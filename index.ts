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
 *
 * Scope-writes (config `defaultScopeWrites`, toggle live with /scopewrites): when on,
 * `writes` mode still confirms write/edit calls that resolve outside the project root.
 *
 * Confirmation timing: all confirmations are asked inline in the `tool_call` hook, and the
 * gated tools are re-registered with executionMode "sequential" so each call in a batched
 * assistant message executes right after its own acceptance (the default parallel path would
 * defer all executions until the last acceptance). The TUI starts bash's elapsed timer on
 * `tool_execution_start`, which fires before the gate, so the re-registered bash tool records
 * the real start time when `execute` begins (after the gate) and the renderers overwrite the
 * timer state with it, keeping the reported "Took" duration accurate.
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

// Re-register the builtin bash tool with two changes:
//   - executionMode "sequential": each batched call runs right after its acceptance.
//   - accurate "Took" time: the TUI sets state.startedAt on the first render after
//     tool_execution_start, which fires before the confirmation gate. execute() only runs
//     after the gate, so we record the real start there and overwrite state.startedAt in
//     the renderers.
function registerSequentialBashTool(pi: ExtensionAPI) {
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
    executionMode: "sequential",
    async execute(toolCallId: string, params: any, signal: any, onUpdate: any, ctx: any) {
      actualStartTimes.set(toolCallId, Date.now());
      try {
        return await base.execute(toolCallId, params, signal, onUpdate, ctx);
      } finally {
        // Keep the entry briefly for the final render, then drop it.
        setTimeout(() => actualStartTimes.delete(toolCallId), 60_000).unref?.();
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

// Re-register the builtin edit tool with one rendering tweak: while the call is
// awaiting confirmation (diff preview computed but tool not yet executed), keep
// the grey pending header background instead of switching to the success color.
function registerPendingAwareEditTool(pi: ExtensionAPI) {
  const base = createEditToolDefinition(process.cwd());
  const isCleanPreview = (component: any): boolean =>
    component?.preview && !("error" in component.preview);

  pi.registerTool({
    ...base,
    // Force the sequential tool-execution path in agent-loop so each edit in a
    // batched assistant message runs right after its confirmation is accepted,
    // instead of all executions being deferred until the last acceptance.
    executionMode: "sequential",
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

  registerPendingAwareEditTool(pi);
  registerSequentialBashTool(pi);

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

    // Seed scope-writes from config, then let any persisted session toggle win.
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

  pi.on("tool_call", async (event, ctx) => decide(event.toolName, event.input, ctx));
}
