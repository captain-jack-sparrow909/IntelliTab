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
    const after = [afterCursor, ...followingLines].join("\n");

    return {
        before,
        after,
        language: document.languageId,
    };
}
