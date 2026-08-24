/**
 * Confirm All Writes Extension (pi-nolo)
 *
 * Gates write, edit, and bash tools behind user confirmation (Enter to allow, Escape to block).
 * Read-safe bash commands are auto-approved: a command is safe when every segment (split on |, &&,
 * ||, ;) starts with a known safe prefix and the command contains no stdout redirects or unsafe
 * constructs. Two layers of dangerous-pattern checks are applied:
 *   global  -- checked on the full command string (backticks, $(), rm, sudo, eval, source)
 *   segment -- checked per segment (sh/bash as commands, find -exec/-delete, system() calls)
 * Stderr redirects such as 2>/dev/null are allowed. Both pattern sets are configurable.
 * Standalone literal assignments (D=/path) are safe segments and $D/${D} references are expanded
 * before prefix matching. Command substitutions $(...) are validated recursively: safe inner
 * commands are replaced with an inert placeholder; unsafe ones fall through to confirmation.
 * cd <literal-dir> is tracked so relative ./x command words resolve to absolute paths before
 * prefix matching: always across &&, and across ;/newlines when the directory is fs-verified
 * at check time (a verified cd cannot fail). | and || always invalidate. Bare newlines
 * separate commands like `;`.
 *
 * YOLO modes (toggle with /yolo or the configured shortcut, default ctrl+y):
 *   off        — default: confirm all writes/edits/bash (safe bash commands auto-approved)
 *   writes     — auto-allow all write/edit; bash still follows safe-prefix rules
 *   full       — auto-allow everything: write, edit, and all bash commands
 * A fresh session starts in config `defaultYoloMode` (default off); a mode persisted in the
 * session history overrides it, so /reload keeps the live mode.
 *
 * Scope-writes (config `defaultScopeWrites`, toggle live with /scopewrites): when on,
 * `writes` mode still confirms write/edit calls that resolve outside the project root.
 *
 * Confirmation timing (pipelined): bash and edit are re-registered so their confirmation
 * happens inside `execute`, not in the `tool_call` gate. On agent-loop's parallel path all
 * gates resolve immediately, so every execute closure starts concurrently; two shared promise
 * chains then coordinate them. The confirm chain shows prompts one at a time in message
 * order. Accepted bash calls run immediately and concurrently (matching pi's default
 * parallel batch behavior); accepted edit calls additionally queue on the exec chain, so
 * they run one at a time in message order. Accepting call N therefore starts (or queues)
 * its execution immediately AND reveals prompt N+1 while earlier calls are still running.
 * Rejected calls fail fast with "Blocked by user" without occupying the exec chain. The TUI
 * starts bash's elapsed timer on `tool_execution_start` (before confirm and queue wait), so
 * the bash wrapper records the real start when its turn on the exec chain begins and the
 * renderers overwrite the timer state with it, keeping "Took" accurate. Write (rarely
 * batched) still confirms in the `tool_call` gate, as does everything in strict
 * non-interactive mode.
 *
 * Strict non-interactive (config `strictNonInteractive`, or env NOLO_STRICT=1/0): when
 * running without a UI (e.g. `pi -p` / --mode json) there is no way to confirm, so by
 * default nothing is gated. With strict mode on, write/edit and unsafe bash commands
 * are instantly blocked instead; safe read-only bash commands still run.
 */

import { createBashToolDefinition, createEditToolDefinition, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { resolve, sep } from "node:path";
import { accessSync, constants, statSync } from "node:fs";
import { loadConfig, DEFAULT_SAFE_PREFIXES, DEFAULT_DANGEROUS_PATTERNS, DEFAULT_SEGMENT_DANGEROUS_PATTERNS } from "./src/config.js";
import { isSafeCommand } from "./src/safety.js";
import type { ToolDecision } from "./src/types.js";

// True when the path is an existing, traversable directory. Lets the safety
// check keep a tracked `cd` directory across `;` boundaries: a cd to a
// verified directory cannot fail, so later segments really run there.
const isExecutableDir = (path: string): boolean => {
  try {
    if (!statSync(path).isDirectory()) return false;
    accessSync(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
};
import {
  createYoloState,
  restoreYoloMode,
  restoreScopeWrites,
  renderStatus,
  cycleYoloMode,
  toggleScopeWrites,
} from "./src/yolo.js";

// Confirmation callback provided by the extension entry point. Late-bound so the
// tool registration helpers can be defined at module scope.
type ConfirmFn = (toolName: string, input: any, ctx: any) => Promise<ToolDecision>;

// Two shared chains pipeline batched tool calls (all execute closures run
// concurrently on agent-loop's parallel path):
//   confirmChain -- prompts appear one at a time, in message order
//   execChain    -- serialized calls (edit) execute one at a time, in message order
// Accepting call N releases its execution and prompt N+1 at once. Bash runs
// concurrently on accept; edit queues on the exec chain. A blocked call throws
// immediately and never occupies the exec chain.
let confirmChain: Promise<unknown> = Promise.resolve();
let execChain: Promise<unknown> = Promise.resolve();

function pipeline<T>(
  confirm: () => Promise<ToolDecision>,
  run: () => Promise<T>,
  serializeExec: boolean,
): Promise<T> {
  const decision = confirmChain.then(confirm, confirm);
  confirmChain = decision.catch(() => undefined);
  return decision.then((d) => {
    if (d?.block) throw new Error(d.reason ?? "Blocked by user");
    if (!serializeExec) return run();
    const result = execChain.then(run, run);
    execChain = result.catch(() => undefined);
    return result;
  });
}

// Re-register the builtin bash tool with two changes:
//   - pipelined confirm via pipeline(); execution starts on accept and runs
//     concurrently with other accepted calls.
//   - accurate "Took" time: the TUI sets state.startedAt on the first render after
//     tool_execution_start, which fires before the confirmation. We record the real
//     start when execution begins and overwrite state.startedAt in the renderers.
function registerPipelinedBashTool(pi: ExtensionAPI, confirm: ConfirmFn) {
  const base = createBashToolDefinition(process.cwd());
  const actualStartTimes = new Map<string, number>();

  const fixStartedAt = (context: any) => {
    const actualStart = actualStartTimes.get(context.toolCallId);
    if (actualStart !== undefined && context.state) {
      context.state.startedAt = actualStart;
    }
  };

  pi.registerTool({
    ...base,
    async execute(toolCallId: string, params: any, signal: any, onUpdate: any, ctx: any) {
      try {
        return await pipeline(
          () => (ctx.hasUI ? confirm("bash", params, ctx) : Promise.resolve(undefined)),
          async () => {
            actualStartTimes.set(toolCallId, Date.now());
            try {
              return await base.execute(toolCallId, params, signal, onUpdate, ctx);
            } finally {
              // Keep the entry briefly for the final render, then drop it.
              setTimeout(() => actualStartTimes.delete(toolCallId), 60_000).unref?.();
            }
          },
          false,
        );
      } catch (err) {
        // Blocked before execution: zero out the timer instead of showing the confirm wait.
        if (!actualStartTimes.has(toolCallId)) actualStartTimes.set(toolCallId, Date.now());
        throw err;
      }
    },
    renderCall(args: any, theme: any, context: any) {
      fixStartedAt(context);
      return base.renderCall!(args, theme, context);
    },
    renderResult(result: any, options: any, theme: any, context: any) {
      fixStartedAt(context);
      return base.renderResult!(result, options, theme, context);
    },
  } as any);
}

// Re-register the builtin edit tool with pipelined confirm + serialized execution,
// and one rendering tweak: while the call is awaiting confirmation (diff preview
// computed but tool not yet executed), keep the grey pending header background
// instead of switching to the success color.
function registerPendingAwareEditTool(pi: ExtensionAPI, confirm: ConfirmFn) {
  const base = createEditToolDefinition(process.cwd());
  const isCleanPreview = (component: any): boolean =>
    component?.preview && !("error" in component.preview);

  pi.registerTool({
    ...base,
    async execute(toolCallId: string, params: any, signal: any, onUpdate: any, ctx: any) {
      return pipeline(
        () => (ctx.hasUI ? confirm("edit", params, ctx) : Promise.resolve(undefined)),
        () => base.execute(toolCallId, params, signal, onUpdate, ctx),
        true,
      );
    },
    renderCall(args: any, theme: any, context: any) {
      const component: any = base.renderCall!(args, theme, context);
      if (!context.state?.noloSettled && isCleanPreview(component)) {
        component.setBgFn((text: string) => theme.bg("toolPendingBg", text));
      }
      return component;
    },
    renderResult(result: any, options: any, theme: any, context: any) {
      if (context.state) context.state.noloSettled = true;
      const callComponent: any = context.state?.callComponent;
      if (callComponent && isCleanPreview(callComponent) && !context.isError) {
        callComponent.setBgFn((text: string) => theme.bg("toolSuccessBg", text));
      }
      return base.renderResult!(result, options, theme, context);
    },
  } as any);
}

export default function (pi: ExtensionAPI) {
  // Resolve the YOLO-cycle shortcut once at load time. registerShortcut takes a
  // literal key, so changing `shortcut` in nolo.json requires /reload to apply.
  const { shortcut } = loadConfig();
  let safePrefixes = DEFAULT_SAFE_PREFIXES;
  let dangerousRegexes = DEFAULT_DANGEROUS_PATTERNS.map((p) => new RegExp(p));
  let segmentDangerousRegexes = DEFAULT_SEGMENT_DANGEROUS_PATTERNS.map((p) => new RegExp(p));
  let projectRoot = process.cwd();
  let strictNonInteractive = loadConfig().strictNonInteractive;
  const yolo = createYoloState();

  // True when scope-writes is on and the path resolves outside the project root.
  const isOutsideRoot = (rawPath: string): boolean => {
    if (!yolo.scopeWrites) return false;
    const resolved = resolve(projectRoot, rawPath);
    return resolved !== projectRoot && !resolved.startsWith(projectRoot + sep);
  };

  // --- Session start: restore mode + reload config ---

  pi.on("session_start", async (_event, ctx) => {
    const config = loadConfig();
    safePrefixes = config.safePrefixes;
    dangerousRegexes = config.dangerousRegexes;
    segmentDangerousRegexes = config.segmentDangerousRegexes;
    strictNonInteractive = config.strictNonInteractive;
    projectRoot = ctx.cwd;

    // Seed from config, then let any persisted session choice win.
    yolo.mode = config.defaultYoloMode;
    yolo.scopeWrites = config.defaultScopeWrites;
    restoreYoloMode(ctx.sessionManager.getEntries(), yolo);
    restoreScopeWrites(ctx.sessionManager.getEntries(), yolo);

    if (ctx.hasUI) {
      ctx.ui.setStatus("nolo", renderStatus(yolo, ctx.ui.theme));
    }

  });

  // --- /yolo command and configured shortcut: cycle through modes ---

  const cycleHandler = async (_argsOrEvent: unknown, ctx: any) => {
    cycleYoloMode(yolo, pi, ctx);
  };

  pi.registerCommand("yolo", {
    description: "Cycle YOLO mode: off → writes-yolo → full-yolo → off",
    handler: cycleHandler,
  });

  pi.registerShortcut(shortcut, {
    description: "Cycle YOLO mode: off → writes-yolo → full-yolo → off",
    handler: async (ctx) => cycleYoloMode(yolo, pi, ctx),
  });

  // --- /scopewrites command: toggle project-root confinement for writes mode ---

  pi.registerCommand("scopewrites", {
    description: "Toggle confirming write/edit outside the project root in writes mode",
    handler: async (_args: unknown, ctx: any) => toggleScopeWrites(yolo, pi, ctx),
  });

  // --- Tool gate ---

  // Runs the gating rules for one tool call, asking the user when needed.
  const decide = async (toolName: string, input: any, ctx: any): Promise<ToolDecision> => {
    // Non-interactive (e.g. `pi -p` / --mode json): no way to confirm.
    // Default: don't gate. In strict mode: instantly block anything that
    // would have required confirmation (write/edit and unsafe bash).
    if (!ctx.hasUI) {
      if (!strictNonInteractive) return undefined;
      if (toolName === "write" || toolName === "edit") {
        return {
          block: true,
          reason: `Blocked by nolo strict non-interactive mode: ${toolName} is not allowed`,
        };
      }
      if (toolName === "bash") {
        const command = input.command as string;
        if (
          isSafeCommand(command, safePrefixes, dangerousRegexes, segmentDangerousRegexes, {
            isExecutableDir,
          })
        ) {
          return undefined;
        }
        return {
          block: true,
          reason: "Blocked by nolo strict non-interactive mode: command is not read-only safe",
        };
      }
      return undefined;
    }

    if (toolName === "write") {
      if (yolo.mode === "full") return undefined;
      if (yolo.mode === "writes" && !isOutsideRoot(input.path as string)) return undefined;

      const path = input.path as string;
      const content = (input.content as string) ?? "";
      const lines = content.split("\n").length;

      const title = yolo.mode === "writes" ? "Write outside project root?" : "Write file?";
      const confirmed = await ctx.ui.confirm(title, `${path} (${lines} lines)`);
      if (!confirmed) return { block: true, reason: "Blocked by user" };

    } else if (toolName === "edit") {
      if (yolo.mode === "full") return undefined;
      if (yolo.mode === "writes" && !isOutsideRoot(input.path as string)) return undefined;

      const title = yolo.mode === "writes" ? "Edit outside project root?" : "Edit file?";
      const confirmed = await ctx.ui.confirm(title, input.path as string);
      if (!confirmed) return { block: true, reason: "Blocked by user" };

    } else if (toolName === "bash") {
      if (yolo.mode === "full") return undefined;

      const command = input.command as string;
      if (
        isSafeCommand(command, safePrefixes, dangerousRegexes, segmentDangerousRegexes, {
          isExecutableDir,
        })
      ) {
        return undefined;
      }

      const firstLine = command.split("\n")[0];
      const preview = command.includes("\n") ? `${firstLine}...` : firstLine;
      const confirmed = await ctx.ui.confirm("Run command?", preview);
      if (!confirmed) return { block: true, reason: "Blocked by user" };
    }

    return undefined;
  };

  registerPendingAwareEditTool(pi, decide);
  registerPipelinedBashTool(pi, decide);

  pi.on("tool_call", async (event, ctx) => {
    // With a UI, bash and edit confirm inside their execute wrappers (pipelined);
    // the gate must not prompt for them or it would serialize prompts behind the
    // prepare loop. Without a UI, decide() only applies strict-mode blocking.
    if (ctx.hasUI && (event.toolName === "bash" || event.toolName === "edit")) return undefined;
    return decide(event.toolName, event.input, ctx);
  });
}
