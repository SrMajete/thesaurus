"""Shared subprocess lifecycle for tools that spawn child processes.

Handles creation, timeout, safe termination, and output formatting in
one place so ``RunBashTool`` and ``RunPythonTool`` stay thin wrappers.
"""

import asyncio

_MAX_OUTPUT_CHARS = 50_000


async def run_subprocess(*args: str, timeout: int, shell: bool = False) -> str:
    """Run a subprocess, enforce a timeout, and return formatted output.

    Parameters
    ----------
    *args:
        When ``shell=True``, a single command string.
        When ``shell=False``, the executable and its arguments.
    timeout:
        Maximum seconds before the process is killed.
    shell:
        If True, run via the system shell (``create_subprocess_shell``).
        If False, run directly (``create_subprocess_exec``).
    """
    try:
        create = asyncio.create_subprocess_shell if shell else asyncio.create_subprocess_exec
        process = await create(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )

    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass  # process already exited between timeout and kill
        await process.wait()
        return f"Error: timed out after {timeout} seconds."

    except OSError as e:
        return f"Error: {e}"

    return _format_output(stdout, stderr, process.returncode)


def _format_output(stdout: bytes, stderr: bytes, returncode: int) -> str:
    """Format subprocess stdout/stderr into a single string with truncation."""
    output = ""
    if stdout:
        output += stdout.decode(errors="replace")
    if stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += stderr.decode(errors="replace")

    if not output:
        output = "(no output)"

    if len(output) > _MAX_OUTPUT_CHARS:
        marker = f"\n\n... truncated ({len(output)} chars total) ...\n\n"
        half = (_MAX_OUTPUT_CHARS - len(marker)) // 2
        output = output[:half] + marker + output[-half:]

    if returncode != 0:
        output += f"\n(exit code: {returncode})"

    return output
