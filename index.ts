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
 * Confirmation timing: the TUI starts bash's elapsed timer when `tool_execution_start`
 * fires, which happens before the `tool_call` gate runs. Confirming there would inflate the
 * reported "Took" duration. So bash confirmations are asked in `message_end` of the assistant
 * message (before any tool starts) and the decision is cached per toolCallId; the `tool_call`
 * hook then only replays the cached decision. Write/edit (no timer) and tool calls with no
 * cached decision still confirm inline in the `tool_call` hook, after the TUI has rendered
 * the tool call and its pre-rendered edit diff.
 *
 * Strict non-interactive (config `strictNonInteractive`, or env NOLO_STRICT=1/0): when
 * running without a UI (e.g. `pi -p` / --mode json) there is no way to confirm, so by
 * default nothing is gated. With strict mode on, write/edit and unsafe bash commands
 * are instantly blocked instead; safe read-only bash commands still run.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
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
import { registerPreRenderEdit } from "./src/pre-render-edit.js";

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

  // Decisions pre-computed in message_end, keyed by toolCallId. `undefined` value
  // means "allow". Consumed (and removed) by the tool_call hook.
  const pendingDecisions = new Map<string, ToolDecision>();

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

    registerPreRenderEdit(pi, ctx.cwd);
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

  // Bash confirmations are asked up front, while no tool has started executing yet,
  // so the TUI's bash elapsed timer (started on tool_execution_start) only covers
  // real execution time. Write/edit stay in the tool_call hook: they have no timer,
  // and confirming them in message_end would run before the TUI marks the tool call
  // args-complete, hiding the pre-rendered inline edit diff.
  pi.on("message_end", async (event, ctx) => {
    const message = event.message as any;
    if (message?.role !== "assistant" || !Array.isArray(message.content)) return undefined;

    pendingDecisions.clear();
    for (const part of message.content) {
      if (part?.type !== "toolCall" || part.name !== "bash") continue;
      pendingDecisions.set(part.id, await decide(part.name, part.arguments ?? {}, ctx));
    }
    return undefined;
  });

  pi.on("tool_call", async (event, ctx) => {
    if (pendingDecisions.has(event.toolCallId)) {
      const decision = pendingDecisions.get(event.toolCallId);
      pendingDecisions.delete(event.toolCallId);
      return decision;
    }
    return decide(event.toolName, event.input, ctx);
  });
}
