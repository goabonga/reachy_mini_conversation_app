"""Unit tests for the code generation tool."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import TextBlock

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.profiles.linus.github_code import GitHubCodeTool


class TestGitHubCodeToolAttributes:
    """Tests for GitHubCodeTool tool attributes."""

    def test_code_has_correct_name(self) -> None:
        """Test GitHubCodeTool tool has correct name."""
        tool = GitHubCodeTool()
        assert tool.name == "github_code"

    def test_code_has_description(self) -> None:
        """Test GitHubCodeTool tool has description."""
        tool = GitHubCodeTool()
        assert "code" in tool.description.lower()
        assert "claude" in tool.description.lower()

    def test_code_has_parameters_schema(self) -> None:
        """Test GitHubCodeTool tool has correct parameters schema."""
        tool = GitHubCodeTool()
        schema = tool.parameters_schema

        assert schema["type"] == "object"
        assert "question" in schema["properties"]
        assert "language" in schema["properties"]
        assert "repo" in schema["properties"]
        assert "output_files" in schema["properties"]
        assert "context_files" in schema["properties"]
        assert "overwrite" in schema["properties"]
        assert "question" in schema["required"]
        assert "repo" in schema["required"]
        assert "output_files" in schema["required"]

    def test_code_spec(self) -> None:
        """Test GitHubCodeTool tool spec generation."""
        tool = GitHubCodeTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "github_code"


class TestGitHubCodeToolExecution:
    """Tests for GitHubCodeTool tool execution."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_code_missing_api_key(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when API key is missing."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="myrepo",
                    output_files=["src/main.py"],
                )

        assert "error" in result
        assert "ANTHROPIC_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_code_missing_repo(self, mock_deps: ToolDependencies) -> None:
        """Test code returns error when repo is missing."""
        tool = GitHubCodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"

            result = await tool(mock_deps, question="Write hello world", output_files=["src/main.py"])

        assert "error" in result
        assert "Repository" in result["error"]

    @pytest.mark.asyncio
    async def test_code_missing_output_files(self, mock_deps: ToolDependencies) -> None:
        """Test code returns error when output_files is missing."""
        tool = GitHubCodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"

            result = await tool(mock_deps, question="Write hello world", repo="myrepo")

        assert "error" in result
        assert "output file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_code_repo_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when repo doesn't exist."""
        tool = GitHubCodeTool()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="nonexistent",
                    output_files=["src/main.py"],
                )

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_code_file_exists_no_overwrite(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when file exists and overwrite is False."""
        tool = GitHubCodeTool()

        # Create repo directory and file
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# existing content")

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="myrepo",
                    output_files=["existing.py"],
                    overwrite=False,
                )

        assert "error" in result
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_code_path_traversal_attack(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error for path traversal attempt."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Write hello world",
                    repo="myrepo",
                    output_files=["../../../etc/passwd"],
                )

        assert "error" in result
        assert "outside" in result["error"].lower() or "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_code_success_single_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code successfully generates and saves a single file to repo."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_message = MagicMock()
        mock_message.content = [
            TextBlock(type="text", text="```python\ndef hello():\n    print('Hello!')\n```")
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None  # Test default model
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        repo="myrepo",
                        output_files=["src/hello.py"],
                    )

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "src/hello.py"
        # Check file was created
        assert (repo_dir / "src" / "hello.py").exists()

    @pytest.mark.asyncio
    async def test_code_success_multiple_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code successfully generates multiple files."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_message = MagicMock()
        mock_message.content = [
            TextBlock(
                type="text",
                text=(
                    "```file:src/my_tool.py\n"
                    "class MyTool:\n    pass\n"
                    "```\n\n"
                    "```file:tests/test_my_tool.py\n"
                    "def test_my_tool():\n    assert True\n"
                    "```"
                ),
            )
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Create a tool and its tests",
                        repo="myrepo",
                        output_files=["src/my_tool.py", "tests/test_my_tool.py"],
                    )

        assert result["status"] == "success"
        assert len(result["files"]) == 2
        # Check files were created
        assert (repo_dir / "src" / "my_tool.py").exists()
        assert (repo_dir / "tests" / "test_my_tool.py").exists()

    @pytest.mark.asyncio
    async def test_code_with_context_files(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code uses context files as templates."""
        tool = GitHubCodeTool()

        # Create repo directory with context files
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "src").mkdir()
        (repo_dir / "src" / "existing_tool.py").write_text(
            "class ExistingTool:\n    '''Existing tool'''\n    pass\n"
        )

        mock_message = MagicMock()
        mock_message.content = [
            TextBlock(
                type="text",
                text="```file:src/new_tool.py\nclass NewTool:\n    '''New tool'''\n    pass\n```",
            )
        ]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Create a new tool like the existing one",
                        repo="myrepo",
                        output_files=["src/new_tool.py"],
                        context_files=["src/existing_tool.py"],
                    )

        assert result["status"] == "success"
        # Verify context was passed to Claude
        call_args = mock_client.messages.create.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "existing_tool.py" in user_content
        assert "ExistingTool" in user_content

    @pytest.mark.asyncio
    async def test_code_context_file_not_found(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error when context file doesn't exist."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Create a new tool",
                    repo="myrepo",
                    output_files=["src/new_tool.py"],
                    context_files=["src/nonexistent.py"],
                )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_code_context_file_path_traversal(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code returns error for context file path traversal."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                result = await tool(
                    mock_deps,
                    question="Create a new tool",
                    repo="myrepo",
                    output_files=["src/new_tool.py"],
                    context_files=["../../../etc/passwd"],
                )

        assert "error" in result
        assert "Invalid" in result["error"] or "must be within" in result["error"]

    @pytest.mark.asyncio
    async def test_code_authentication_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles authentication error."""
        import anthropic

        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "invalid-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        repo="myrepo",
                        output_files=["src/main.py"],
                    )

        assert "error" in result
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_code_rate_limit_error(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles rate limit error."""
        import anthropic

        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit"}},
        )

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        repo="myrepo",
                        output_files=["src/main.py"],
                    )

        assert "error" in result
        assert "Rate limit" in result["error"]

    @pytest.mark.asyncio
    async def test_code_generic_exception(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles generic exceptions."""
        tool = GitHubCodeTool()

        # Create repo directory
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("Network error")

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write hello world",
                        repo="myrepo",
                        output_files=["src/main.py"],
                    )

        assert "error" in result
        assert "Failed to generate code" in result["error"]


class TestGitHubCodeToolHelpers:
    """Tests for GitHubCodeTool helper methods."""

    def test_parse_generated_files_with_file_format(self) -> None:
        """Test parsing files with ```file:path``` format."""
        tool = GitHubCodeTool()

        response = (
            "```file:src/tool.py\n"
            "class Tool:\n    pass\n"
            "```\n\n"
            "```file:tests/test_tool.py\n"
            "def test():\n    assert True\n"
            "```"
        )
        files = tool._parse_generated_files(response, ["src/tool.py", "tests/test_tool.py"])

        assert len(files) == 2
        assert "src/tool.py" in files
        assert "tests/test_tool.py" in files
        assert "class Tool:" in files["src/tool.py"]
        assert "def test():" in files["tests/test_tool.py"]

    def test_parse_generated_files_single_file_fallback(self) -> None:
        """Test parsing single file with standard code block."""
        tool = GitHubCodeTool()

        response = "```python\ndef hello():\n    print('Hello')\n```"
        files = tool._parse_generated_files(response, ["main.py"])

        assert len(files) == 1
        assert "main.py" in files
        assert "def hello():" in files["main.py"]

    def test_parse_generated_files_no_language_tag(self) -> None:
        """Test parsing single file without language tag."""
        tool = GitHubCodeTool()

        response = "```\ndef hello():\n    print('Hello')\n```"
        files = tool._parse_generated_files(response, ["main.py"])

        assert len(files) == 1
        assert "main.py" in files

    def test_get_extension_python(self) -> None:
        """Test getting extension for Python."""
        tool = GitHubCodeTool()
        assert tool._get_extension("python") == "py"
        assert tool._get_extension("Python") == "py"

    def test_get_extension_javascript(self) -> None:
        """Test getting extension for JavaScript."""
        tool = GitHubCodeTool()
        assert tool._get_extension("javascript") == "js"

    def test_get_extension_typescript(self) -> None:
        """Test getting extension for TypeScript."""
        tool = GitHubCodeTool()
        assert tool._get_extension("typescript") == "ts"

    def test_get_extension_rust(self) -> None:
        """Test getting extension for Rust."""
        tool = GitHubCodeTool()
        assert tool._get_extension("rust") == "rs"

    def test_get_extension_cpp(self) -> None:
        """Test getting extension for C++."""
        tool = GitHubCodeTool()
        assert tool._get_extension("cpp") == "cpp"
        assert tool._get_extension("c++") == "cpp"

    def test_get_extension_unknown(self) -> None:
        """Test getting extension for unknown language."""
        tool = GitHubCodeTool()
        assert tool._get_extension("unknown_lang") == "txt"


class TestGitHubCodeToolOverwrite:
    """Tests for GitHubCodeTool with overwrite functionality."""

    @pytest.fixture
    def mock_deps(self) -> ToolDependencies:
        """Create mock tool dependencies."""
        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_code_overwrite_existing_file(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code can overwrite existing file in repo."""
        tool = GitHubCodeTool()

        # Create repo directory and file
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "existing.py").write_text("# old content")

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="```python\n# new content\n```")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write new content",
                        repo="myrepo",
                        output_files=["existing.py"],
                        overwrite=True,
                    )

        assert result["status"] == "success"
        assert (repo_dir / "existing.py").read_text() == "# new content"

    @pytest.mark.asyncio
    async def test_code_repo_with_slash(self, mock_deps: ToolDependencies, tmp_path: Path) -> None:
        """Test code handles repo name with owner/repo format."""
        tool = GitHubCodeTool()

        # Create repo directory (only the repo name, not owner)
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()

        mock_message = MagicMock()
        mock_message.content = [TextBlock(type="text", text="```python\nprint('hello')\n```")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        with patch("reachy_mini_conversation_app.profiles.linus.github_code.config") as mock_config:
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = None
            with patch("reachy_mini_conversation_app.profiles.linus.github_code.REPOS_DIR", tmp_path):
                with patch(
                    "reachy_mini_conversation_app.profiles.linus.github_code.anthropic.Anthropic",
                    return_value=mock_client,
                ):
                    result = await tool(
                        mock_deps,
                        question="Write hello",
                        repo="owner/myrepo",
                        output_files=["main.py"],
                    )

        assert result["status"] == "success"
        assert result["repo"] == "myrepo"
