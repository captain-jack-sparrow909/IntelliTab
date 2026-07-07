/**
 * Debounce utility.
 *
 * Wraps a function so it only fires after the caller stops invoking
 * it for the specified delay.  Cancels any pending invocation when
 * called again before the delay elapses.
 */

export interface DebouncedFunction<T extends (...args: unknown[]) => void> {
    /** Invoke the debounced function. Resets the timer if called again within the delay. */
    (...args: Parameters<T>): void;
    /** Cancel any pending invocation. */
    cancel(): void;
    /** Immediately invoke the wrapped function if there's a pending invocation. */
    flush(): void;
}

/**
 * Create a debounced version of the given function.
 *
 * @param func - The function to debounce
 * @param delayMs - Minimum milliseconds between invocations
 * @returns A debounced function with `cancel()` and `flush()` methods
 */
export function debounce<T extends (...args: unknown[]) => void>(
    func: T,
    delayMs: number,
): DebouncedFunction<T> {
    let timer: ReturnType<typeof setTimeout> | null = null;

    const debounced = function (...args: Parameters<T>) {
        if (timer !== null) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => {
            timer = null;
            func(...args);
        }, delayMs);
    } as DebouncedFunction<T>;

    debounced.cancel = () => {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
        }
    };

    debounced.flush = () => {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
            func();
        }
    };

    return debounced;
}
