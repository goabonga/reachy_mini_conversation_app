"""Code generation tool using Claude API with context files support."""

import re
import logging
from typing import Any, Dict, List
from pathlib import Path

import anthropic
from anthropic.types import TextBlock

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Directory where repositories are cloned
REPOS_DIR = Path.home() / "reachy_repos"

# Maximum file size to read as context (100KB)
MAX_CONTEXT_FILE_SIZE = 100 * 1024


class GitHubCodeTool(Tool):
    """Generate code using Claude API and save to a repository."""

    name = "github_code"
    description = (
        "Generate code using Claude AI and save it directly to a repository. "
        "Can read existing files as context/templates and generate multiple files at once. "
        "Use this to create new tools, tests, or any code following existing patterns."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The coding question or task to solve",
            },
            "language": {
                "type": "string",
                "description": "Programming language (e.g., python, javascript, rust). Defaults to python.",
            },
            "repo": {
                "type": "string",
                "description": "Repository name to write to (folder in ~/reachy_repos/).",
            },
            "output_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of output file paths to generate (e.g., ['src/tools/my_tool.py', 'tests/test_my_tool.py']). "
                    "Claude will generate content for each file."
                ),
            },
            "context_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of existing files to read as context/templates (e.g., ['src/tools/other_tool.py']). "
                    "These files will be provided to Claude as examples to follow."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite existing files. Default: false",
            },
        },
        "required": ["question", "repo", "output_files"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Generate code using Claude API."""
        question = kwargs.get("question", "")
        language = kwargs.get("language", "python")
        repo = kwargs.get("repo", "")
        output_files: List[str] = kwargs.get("output_files", [])
        context_files: List[str] = kwargs.get("context_files", [])
        overwrite = kwargs.get("overwrite", False)

        logger.info(
            f"Tool call: github_code - question='{question[:50]}...', "
            f"repo={repo}, outputs={len(output_files)}, contexts={len(context_files)}"
        )

        # Check for API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            return {
                "error": "ANTHROPIC_API_KEY is not configured. "
                "Please set it in your .env file to use code generation."
            }

        # Validate required parameters
        if not repo:
            return {"error": "Repository name is required."}
        if not output_files:
            return {"error": "At least one output file path is required."}

        # Check repo exists
        local_name = repo.split("/")[-1] if "/" in repo else repo
        repo_dir = REPOS_DIR / local_name
        if not repo_dir.exists():
            return {
                "error": f"Repository not found: {local_name}",
                "hint": "Use github_clone to clone the repository first.",
            }

        # Validate output paths and check for existing files
        for out_path in output_files:
            dest_file = repo_dir / out_path
            # Validate path is within repo
            try:
                dest_file.resolve().relative_to(repo_dir.resolve())
            except ValueError:
                return {"error": f"Invalid path: {out_path} - cannot write outside the repository."}

            if dest_file.exists() and not overwrite:
                return {
                    "error": f"File already exists: {out_path}",
                    "hint": "Set overwrite=true to replace existing files.",
                }

        # Read context files
        context_contents: Dict[str, str] = {}
        for ctx_path in context_files:
            ctx_file = repo_dir / ctx_path
            try:
                ctx_file.resolve().relative_to(repo_dir.resolve())
            except ValueError:
                return {"error": f"Invalid context path: {ctx_path} - must be within the repository."}

            if not ctx_file.exists():
                return {"error": f"Context file not found: {ctx_path}"}

            if ctx_file.stat().st_size > MAX_CONTEXT_FILE_SIZE:
                return {"error": f"Context file too large: {ctx_path} (max {MAX_CONTEXT_FILE_SIZE // 1024}KB)"}

            try:
                context_contents[ctx_path] = ctx_file.read_text(encoding="utf-8")
            except Exception as e:
                return {"error": f"Failed to read context file {ctx_path}: {e}"}

        # Build the prompt for Claude
        system_prompt = self._build_system_prompt(output_files)
        user_prompt = self._build_user_prompt(question, language, output_files, context_contents)

        try:
            # Call Claude API
            client = anthropic.Anthropic(api_key=api_key)
            model = config.ANTHROPIC_MODEL or "claude-sonnet-4-20250514"

            message = client.messages.create(
                model=model,
                max_tokens=8192,  # Increased for multiple files
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract response
            first_block = message.content[0]
            response_text = first_block.text if isinstance(first_block, TextBlock) else str(first_block)

            # Parse the generated files from response
            generated_files = self._parse_generated_files(response_text, output_files)

            if not generated_files:
                return {
                    "error": "Failed to parse generated code from Claude's response.",
                    "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                }

            # Write all generated files
            written_files = []
            for file_path, content in generated_files.items():
                dest_file = repo_dir / file_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(content, encoding="utf-8")
                written_files.append({
                    "path": file_path,
                    "lines": len(content.splitlines()),
                })
                logger.info(f"Code saved to {dest_file}")

            return {
                "status": "success",
                "message": f"Generated {len(written_files)} file(s) in {local_name}",
                "repo": local_name,
                "files": written_files,
                "hint": "Use github_add to stage the files, then github_commit to commit.",
            }

        except anthropic.AuthenticationError:
            return {"error": "Invalid ANTHROPIC_API_KEY. Please check your API key."}
        except anthropic.RateLimitError:
            return {"error": "Rate limit exceeded. Please try again later."}
        except Exception as e:
            logger.exception(f"Error generating code: {e}")
            return {"error": f"Failed to generate code: {str(e)}"}

    def _build_system_prompt(self, output_files: List[str]) -> str:
        """Build the system prompt for Claude."""
        return (
            "You are a skilled programmer. Generate clean, well-documented code.\n\n"
            "IMPORTANT: You must output code for EACH requested file in the following format:\n\n"
            "```file:path/to/file.py\n"
            "# code content here\n"
            "```\n\n"
            f"You need to generate {len(output_files)} file(s):\n"
            + "\n".join(f"- {f}" for f in output_files) + "\n\n"
            "Each file block MUST start with ```file: followed by the exact file path.\n"
            "Follow existing patterns and conventions from context files if provided.\n"
            "Include proper docstrings, type hints, and comments."
        )

    def _build_user_prompt(
        self,
        question: str,
        language: str,
        output_files: List[str],
        context_contents: Dict[str, str],
    ) -> str:
        """Build the user prompt with context files."""
        parts = []

        # Add context files first
        if context_contents:
            parts.append("## Context Files (use as templates/examples):\n")
            for path, content in context_contents.items():
                parts.append(f"### {path}\n```{language}\n{content}\n```\n")
            parts.append("\n")

        # Add the task
        parts.append(f"## Task:\n{question}\n\n")

        # Add output files reminder
        parts.append(f"## Required Output Files ({language}):\n")
        for f in output_files:
            parts.append(f"- {f}\n")

        parts.append(
            "\nGenerate complete, working code for each file. "
            "Follow the patterns from the context files if provided."
        )

        return "".join(parts)

    def _parse_generated_files(self, response: str, expected_files: List[str]) -> Dict[str, str]:
        """Parse multiple files from Claude's response."""
        files: Dict[str, str] = {}

        # Pattern to match ```file:path\n...content...\n```
        pattern = r"```file:([^\n]+)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for file_path, content in matches:
            file_path = file_path.strip()
            content = content.strip()
            files[file_path] = content

        # If no files found with the file: format, try to extract single file
        if not files and len(expected_files) == 1:
            # Fall back to simple code block extraction
            simple_pattern = r"```(?:\w+)?\n(.*?)```"
            simple_matches = re.findall(simple_pattern, response, re.DOTALL)
            if simple_matches:
                files[expected_files[0]] = simple_matches[0].strip()

        return files

    def _get_extension(self, language: str) -> str:
        """Get file extension for a language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "rust": "rs",
            "go": "go",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "ruby": "rb",
            "php": "php",
            "swift": "swift",
            "kotlin": "kt",
            "scala": "scala",
            "shell": "sh",
            "bash": "sh",
            "sql": "sql",
            "html": "html",
            "css": "css",
        }
        return extensions.get(language.lower(), "txt")
