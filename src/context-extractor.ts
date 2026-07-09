/**
 * Adaptive structured context for FIM + intent completions.
 *
 * Accuracy-focused (without huge prefill):
 * - Always include top-of-file imports / package header
 * - Prefer the enclosing function/class over raw N lines
 * - Scale window size by completion mode (fast mid-line vs rich intent)
 */

import * as vscode from "vscode";

/** How the model should treat this request. */
export type CompletionMode = "fim" | "intent";

export interface DocumentContext {
    before: string;
    after: string;
    language: string;
    intent?: string;
    /** Dual-policy mode for the backend. */
    mode: CompletionMode;
    /** Debug / metrics. */
    meta?: {
        linesBefore: number;
        linesAfter: number;
        usedImports: boolean;
        usedScope: boolean;
    };
}

export interface ContextBudgets {
    /** Soft cap for normal FIM mid-line. */
    fimBefore: number;
    fimAfter: number;
    /** Richer window for intent / empty-body fill. */
    intentBefore: number;
    intentAfter: number;
}

const DEFAULT_BUDGETS: ContextBudgets = {
    fimBefore: 50,
    fimAfter: 12,
    intentBefore: 100,
    intentAfter: 25,
};

// --- public API -------------------------------------------------------------

/**
 * Extract adaptive context around the cursor.
 *
 * @param maxBefore - user setting upper bound (not a fixed window)
 * @param maxAfter  - user setting upper bound
 */
export function extractContext(
    document: vscode.TextDocument,
    position: vscode.Position,
    maxBefore: number = 60,
    maxAfter: number = 15,
    intent?: string,
    budgets: ContextBudgets = DEFAULT_BUDGETS,
): DocumentContext {
    const isIntent = !!intent;
    const mode: CompletionMode = isIntent ? "intent" : "fim";

    // Adaptive line counts — never exceed user max settings.
    let linesBefore = isIntent
        ? Math.min(maxBefore, budgets.intentBefore)
        : Math.min(maxBefore, budgets.fimBefore);
    let linesAfter = isIntent
        ? Math.min(maxAfter, budgets.intentAfter)
        : Math.min(maxAfter, budgets.fimAfter);

    // Mid-identifier (e.g. `obj.fooBa|`) → tighter window, faster prefill.
    if (!isIntent && isMidIdentifier(document, position)) {
        linesBefore = Math.min(linesBefore, 30);
        linesAfter = Math.min(linesAfter, 8);
    }

    // After `.` `(` → slightly more local context for member/call completion.
    if (!isIntent && isAfterTriggerChar(document, position)) {
        linesBefore = Math.min(Math.max(linesBefore, 40), maxBefore);
    }

    const lineCount = document.lineCount;
    const currentLine = document.lineAt(position);
    const beforeCursor = currentLine.text.substring(0, position.character);
    const afterCursor = currentLine.text.substring(position.character);

    // --- imports (always high value, small) ---
    const imports = extractImports(document, Math.min(80, position.line));
    const usedImports = imports.length > 0;

    // --- enclosing scope (function / class / method) ---
    const scope = findEnclosingScope(document, position.line);
    const usedScope = scope !== null;

    // Local window before cursor
    const localStart = Math.max(0, position.line - linesBefore);
    let prefixStart = localStart;

    // If we found a scope start, prefer starting there (within a sane limit).
    if (scope && scope.startLine < position.line) {
        const scopeSpan = position.line - scope.startLine;
        if (scopeSpan <= linesBefore + 40) {
            prefixStart = scope.startLine;
        } else {
            // Scope too large — keep a slice of it plus local window.
            prefixStart = Math.max(scope.startLine, position.line - linesBefore);
        }
    }

    const precedingLines: string[] = [];
    for (let i = prefixStart; i < position.line; i++) {
        precedingLines.push(document.lineAt(i).text);
    }

    // After-cursor window
    const endLine = Math.min(lineCount - 1, position.line + linesAfter);
    const followingLines: string[] = [];
    for (let i = position.line + 1; i <= endLine; i++) {
        followingLines.push(document.lineAt(i).text);
    }

    // Stitch: imports (if not already in prefix) + local/scope prefix + cursor line.
    const prefixBody = [...precedingLines, beforeCursor].join("\n");
    let before: string;
    if (usedImports && !prefixAlreadyHasImports(prefixBody, imports)) {
        // Keep imports compact; blank line separator.
        before = imports.join("\n") + "\n\n" + prefixBody;
    } else {
        before = prefixBody;
    }

    let after = [afterCursor, ...followingLines].join("\n");

    // Cursor right before `{` `(` `[` — strip so FIM fills the body, not the brace.
    if (after.startsWith("{") || after.startsWith("(") || after.startsWith("[")) {
        after = after.slice(1);
    }

    // Hard character caps (prefill insurance).
    before = truncateEnd(before, isIntent ? 4500 : 3200);
    after = truncateStart(after, isIntent ? 1200 : 600);

    return {
        before,
        after,
        language: document.languageId,
        intent,
        mode,
        meta: {
            linesBefore: position.line - prefixStart,
            linesAfter: endLine - position.line,
            usedImports,
            usedScope,
        },
    };
}

/**
 * Detect Copilot-style intent: comment→code or empty block body.
 *
 * IMPORTANT: Do NOT treat mid-line expression completions as intent.
 *   `const sub = (a, b) => |`  → normal FIM (fill expression)
 *   `const sub = (a, b) => {` + empty body → intent / block fill
 *
 * Misclassifying `=> |` as intent caused multi-line garbage like `a + b b;`.
 */
export function detectIntent(
    document: vscode.TextDocument,
    position: vscode.Position,
): string | undefined {
    const currentLine = document.lineAt(position.line).text;
    const beforeCursor = currentLine.substring(0, position.character);

    // 1. Comment on the current line.
    const commentMatch = beforeCursor.match(/(?:\/\/|#|"""|'''|\/\*)\s*(.+?)\s*$/);
    if (commentMatch && commentMatch[1].length >= 3) {
        return commentMatch[1];
    }

    // 2. Standalone comment on the line above, cursor on empty/indent line.
    if (position.line > 0 && beforeCursor.trim().length === 0) {
        const prevLine = document.lineAt(position.line - 1).text;
        const prevComment = prevLine.match(/(?:\/\/|#|"""|'''|\/\*)\s*(.+?)\s*$/);
        if (prevComment && prevComment[1].length >= 3) {
            return prevComment[1];
        }
    }

    // 3. Signature that *opens a block* and has no body yet on this line.
    //    Requires `{` or Python `:` — bare `=>` is expression FIM, not intent.
    const blockSig = beforeCursor.match(
        /(?:function\s+[\w$]+\s*\([^)]*\)\s*\{\s*$|def\s+[\w$]+\s*\([^)]*\)\s*:\s*$|(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?:=>|->)\s*\{\s*$|[\w$]+\s*\([^)]*\)\s*(?:=>|->)\s*\{\s*$)/,
    );
    if (blockSig) {
        return `Implement the body of:\n${beforeCursor.trim()}`;
    }

    // 4. Empty function / block body: cursor on blank line inside `{ ... }`.
    if (beforeCursor.trim().length === 0) {
        const emptyBody = detectEmptyBodyIntent(document, position);
        if (emptyBody) {
            return emptyBody;
        }
    }

    return undefined;
}

// --- helpers ----------------------------------------------------------------

function isMidIdentifier(document: vscode.TextDocument, position: vscode.Position): boolean {
    const line = document.lineAt(position.line).text;
    const before = line.substring(0, position.character);
    // Cursor sits after identifier characters (completing a name/member).
    return /[A-Za-z0-9_$]$/.test(before);
}

function isAfterTriggerChar(document: vscode.TextDocument, position: vscode.Position): boolean {
    if (position.character === 0) {
        return false;
    }
    const ch = document.lineAt(position.line).text.charAt(position.character - 1);
    return ch === "." || ch === "(" || ch === "[" || ch === " ";
}

/** Collect import/using/include/package lines from the file header. */
function extractImports(document: vscode.TextDocument, upToLine: number): string[] {
    const out: string[] = [];
    const limit = Math.min(document.lineCount, Math.max(upToLine, 0), 100);
    let seenCode = false;

    for (let i = 0; i < limit; i++) {
        const text = document.lineAt(i).text;
        const trimmed = text.trim();

        if (!trimmed) {
            if (out.length > 0) {
                // allow blank lines between import groups
                continue;
            }
            continue;
        }

        // Shebang / file-level comments before imports
        if (
            !seenCode &&
            (trimmed.startsWith("#!") ||
                trimmed.startsWith("//") ||
                trimmed.startsWith("#") ||
                trimmed.startsWith("/*") ||
                trimmed.startsWith("*") ||
                trimmed.startsWith('"""') ||
                trimmed.startsWith("'''"))
        ) {
            continue;
        }

        if (isImportLine(trimmed)) {
            out.push(text);
            seenCode = true;
            continue;
        }

        // Stop at first real code after we've started (or immediately if not import).
        if (out.length > 0) {
            break;
        }
        // No imports yet but non-import code — give up.
        if (!isImportLine(trimmed)) {
            break;
        }
    }

    // Cap import block size.
    if (out.length > 40) {
        return out.slice(0, 40);
    }
    return out;
}

function isImportLine(trimmed: string): boolean {
    return (
        /^(import|from|export\s+[\{\*]|export\s+\{|require\s*\(|using\s+|include\s+[#<]|package\s+|use\s+)/.test(
            trimmed,
        ) ||
        /^#include\b/.test(trimmed) ||
        /^@use\b|^@import\b/.test(trimmed)
    );
}

function prefixAlreadyHasImports(prefix: string, imports: string[]): boolean {
    if (imports.length === 0) {
        return true;
    }
    // If the first import line appears in prefix, assume imports already included.
    const first = imports[0].trim();
    return first.length > 0 && prefix.includes(first);
}

interface ScopeInfo {
    startLine: number;
    header: string;
}

/**
 * Walk upward to find the nearest function / class / method header.
 * Indentation + keyword heuristic (no full parser — fast enough for IDE use).
 */
function findEnclosingScope(
    document: vscode.TextDocument,
    cursorLine: number,
): ScopeInfo | null {
    const cursorIndent = leadingIndent(document.lineAt(cursorLine).text);

    for (let i = cursorLine; i >= 0; i--) {
        const text = document.lineAt(i).text;
        const trimmed = text.trim();
        if (!trimmed) {
            continue;
        }
        const indent = leadingIndent(text);

        // Only consider headers at indent <= cursor indent (outer or same).
        if (indent > cursorIndent && i !== cursorLine) {
            continue;
        }

        if (isScopeHeader(trimmed)) {
            // Prefer headers strictly above the cursor line.
            if (i < cursorLine || (i === cursorLine && isScopeHeader(trimmed))) {
                return { startLine: i, header: trimmed };
            }
        }

        // Don't scan forever on huge files.
        if (cursorLine - i > 400) {
            break;
        }
    }
    return null;
}

function isScopeHeader(trimmed: string): boolean {
    return (
        /^(export\s+)?(async\s+)?function\b/.test(trimmed) ||
        /^(export\s+)?(default\s+)?class\b/.test(trimmed) ||
        /^(export\s+)?(async\s+)?(const|let|var)\s+[\w$]+\s*=\s*(async\s*)?\(/.test(trimmed) ||
        /^(export\s+)?(async\s+)?(const|let|var)\s+[\w$]+\s*=\s*(async\s*)?function\b/.test(
            trimmed,
        ) ||
        /^(public|private|protected|static|async|override|export)\b/.test(trimmed) ||
        /^def\s+\w+/.test(trimmed) ||
        /^class\s+\w+/.test(trimmed) ||
        /^fn\s+\w+/.test(trimmed) ||
        /^func\s+\w+/.test(trimmed) ||
        /^impl\b/.test(trimmed) ||
        /^interface\b|^type\s+\w+\s*=|^enum\b/.test(trimmed) ||
        // method style: name(...) { or name(...) ->
        /^[\w$]+\s*\([^)]*\)\s*(\{|:|=>|->)/.test(trimmed)
    );
}

function leadingIndent(line: string): number {
    const m = line.match(/^[\t ]*/);
    return m ? m[0].length : 0;
}

/**
 * Empty body inside braces / Python def block → intent to fill implementation.
 *
 * Only fires when the block body has NO real statements yet. If the user already
 * has partial/broken code inside, return undefined so normal FIM can fix a line
 * instead of dumping another full (conflicting) body.
 */
function detectEmptyBodyIntent(
    document: vscode.TextDocument,
    position: vscode.Position,
): string | undefined {
    // Find nearest opening header above the cursor.
    let headerLine = -1;
    let headerText = "";
    for (let i = position.line; i >= Math.max(0, position.line - 12); i--) {
        const t = document.lineAt(i).text.trim();
        if (!t) {
            continue;
        }
        if (
            /\{$/.test(t) ||
            /:\s*$/.test(t) ||
            /=>\s*\{\s*$/.test(t) ||
            /function\s+[\w$]+\s*\([^)]*\)\s*\{\s*$/.test(t)
        ) {
            headerLine = i;
            headerText = t;
            break;
        }
        // Stop if we hit real code that isn't a pure signature opener.
        if (i < position.line) {
            break;
        }
    }
    if (headerLine < 0) {
        return undefined;
    }

    // Find matching closer (first `}` at indent <= header, or end).
    const headerIndent = leadingIndent(document.lineAt(headerLine).text);
    let closerLine = -1;
    for (let i = headerLine + 1; i < Math.min(document.lineCount, headerLine + 40); i++) {
        const line = document.lineAt(i).text;
        const t = line.trim();
        if (!t) {
            continue;
        }
        const ind = leadingIndent(line);
        if (t.startsWith("}") && ind <= headerIndent) {
            closerLine = i;
            break;
        }
    }

    // Scan body lines (exclusive of header/closer) for existing statements.
    const bodyStart = headerLine + 1;
    const bodyEnd = closerLine > 0 ? closerLine : Math.min(document.lineCount, position.line + 1);
    for (let i = bodyStart; i < bodyEnd; i++) {
        const t = document.lineAt(i).text.trim();
        if (!t) {
            continue;
        }
        // Placeholders / comments only → still "empty" for intent purposes.
        if (
            t.startsWith("//") ||
            t.startsWith("#") ||
            t.startsWith("/*") ||
            /your code here/i.test(t) ||
            /^pass$/.test(t) ||
            /^\.\.\.$/.test(t)
        ) {
            continue;
        }
        // Real code already present → do NOT intent-fill (would duplicate).
        return undefined;
    }

    // Cursor must sit inside the empty body region.
    if (position.line < bodyStart || (closerLine > 0 && position.line >= closerLine)) {
        return undefined;
    }

    let header = headerText;
    if (headerLine > 0) {
        const above = document.lineAt(headerLine - 1).text.trim();
        if (above.startsWith("@") || above.startsWith("//") || above.startsWith("#")) {
            header = above + "\n" + header;
        }
    }
    return `Write only the function body for:\n${header}\nDo not repeat the signature. Do not use placeholders.`;
}

function truncateEnd(text: string, maxChars: number): string {
    if (text.length <= maxChars) {
        return text;
    }
    return text.slice(text.length - maxChars);
}

function truncateStart(text: string, maxChars: number): string {
    if (text.length <= maxChars) {
        return text;
    }
    return text.slice(0, maxChars);
}
