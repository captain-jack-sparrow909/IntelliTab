/**
 * Context extractor.
 *
 * Reads lines from the active document around the cursor position
 * and builds the structured context payload sent to the MLX backend.
 */

import * as vscode from "vscode";

export interface DocumentContext {
    /** Lines before the cursor (current line prefix + preceding lines) */
    before: string;
    /** Lines after the cursor (current line suffix + following lines) */
    after: string;
    /** Current programming language identifier */
    language: string;
    /** When set, generate code from this description/comment (Copilot-style) */
    intent?: string;
}

/**
 * Extract a context window around the cursor position.
 *
 * @param document - The active text document
 * @param position - The cursor position
 * @param linesBefore - Number of lines to capture before cursor (default 150)
 * @param linesAfter - Number of lines to capture after cursor (default 35)
 * @returns Structured context for FIM prompting
 */
export function extractContext(
    document: vscode.TextDocument,
    position: vscode.Position,
    linesBefore: number = 150,
    linesAfter: number = 35,
): DocumentContext {
    const lineCount = document.lineCount;

    // Current line text split at cursor
    const currentLine = document.lineAt(position);
    const beforeCursor = currentLine.text.substring(0, position.character);
    const afterCursor = currentLine.text.substring(position.character);

    // Lines before current line
    const startLine = Math.max(0, position.line - linesBefore);
    const precedingLines: string[] = [];
    for (let i = startLine; i < position.line; i++) {
        precedingLines.push(document.lineAt(i).text);
    }

    // Lines after current line
    const endLine = Math.min(lineCount - 1, position.line + linesAfter);
    const followingLines: string[] = [];
    for (let i = position.line + 1; i <= endLine; i++) {
        followingLines.push(document.lineAt(i).text);
    }

    // Assemble before: preceding lines + current line prefix
    const before = [...precedingLines, beforeCursor].join("\n");

    // Assemble after: current line suffix + following lines
    let after = [afterCursor, ...followingLines].join("\n");

    // When the cursor sits right before an opening bracket (e.g. the cursor is
    // at `const add = (a, b) => │{` and the file continues with `}`), the model
    // gets confused and re-emits the signature. Strip a single leading bracket
    // from `after` so the model completes the *body* instead.
    if (after.startsWith("{") || after.startsWith("(") || after.startsWith("[")) {
        after = after.slice(1);
    }

    return {
        before,
        after,
        language: document.languageId,
    };
}

/**
 * Detect a "Copilot-style" intent: the user wrote a comment describing code,
 * or a function signature without a body, and we want the model to generate
 * the implementation. Returns the intent text, or undefined if this is a
 * normal mid-line completion.
 */
export function detectIntent(
    document: vscode.TextDocument,
    position: vscode.Position,
): string | undefined {
    const lineCount = document.lineCount;
    const currentLine = document.lineAt(position.line).text;
    const beforeCursor = currentLine.substring(0, position.character);

    // 1. Comment on the current line (e.g. "// fetch user data").
    const commentMatch = beforeCursor.match(/(?:\/\/|#|"""|'''|\/\*)\s*(.+?)\s*$/);
    if (commentMatch && commentMatch[1].length >= 3) {
        return commentMatch[1];
    }

    // 2. A standalone comment line immediately above the cursor.
    if (position.line > 0) {
        const prevLine = document.lineAt(position.line - 1).text;
        const prevComment = prevLine.match(/(?:\/\/|#|"""|'''|\/\*)\s*(.+?)\s*$/);
        if (prevComment && prevComment[1].length >= 3 && beforeCursor.trim().length === 0) {
            return prevComment[1];
        }
    }

    // 3. A function/method signature without a body on the current line
    //    (e.g. "function add(a, b) {" or "def add(a, b):" with no statements).
    const sigMatch = beforeCursor.match(
        /(function\s+[\w$]+\s*\([^)]*\)\s*\{?\s*$|def\s+[\w$]+\s*\([^)]*\)\s*:\s*$|[\w$]+\s*\([^)]*\)\s*(?:=>|->)\s*$)/,
    );
    if (sigMatch) {
        return `Implement the following:\n${beforeCursor.trim()}`;
    }

    return undefined;
}

