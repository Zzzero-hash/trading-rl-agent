#!/usr/bin/env python3
"""
Documentation Cleanup Script

This script identifies and helps clean up redundant documentation files
in the Trading RL Agent project.
"""

import hashlib
import os
import re
import shutil
from pathlib import Path


class DocumentationCleanup:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "docs"
        self.redundant_files = []
        self.consolidation_map = {}

    def find_markdown_files(self) -> list[Path]:
        """Find all markdown files in the project."""
        markdown_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["venv", "__pycache__", "_build"]]

            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(Path(root) / file)
        return markdown_files

    def analyze_file_content(self, file_path: Path) -> dict:
        """Analyze a markdown file for content and metadata."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Calculate file hash for duplicate detection
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Extract title and basic info
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem

            # Count lines and estimate size
            lines = content.split("\n")
            word_count = len(content.split())

            return {
                "path": file_path,
                "title": title,
                "content_hash": content_hash,
                "lines": len(lines),
                "word_count": word_count,
                "size_kb": len(content) / 1024,
                "content": content[:500],  # First 500 chars for similarity detection
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def find_duplicates(self, files: list[dict]) -> list[list[dict]]:
        """Find files with identical content."""
        hash_groups = {}
        for file_info in files:
            if file_info:
                content_hash = file_info["content_hash"]
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(file_info)

        # Return groups with more than one file
        return [group for group in hash_groups.values() if len(group) > 1]

    def find_similar_files(self, files: list[dict]) -> list[list[dict]]:
        """Find files with similar titles or content."""
        similar_groups = []
        processed = set()

        for i, file1 in enumerate(files):
            if not file1 or i in processed:
                continue

            similar_group = [file1]
            processed.add(i)

            for j, file2 in enumerate(files[i + 1 :], i + 1):
                if not file2 or j in processed:
                    continue

                # Check for similar titles
                title1 = file1["title"].lower().replace("_", " ").replace("-", " ")
                title2 = file2["title"].lower().replace("_", " ").replace("-", " ")

                # Simple similarity check
                if title1 in title2 or title2 in title1 or self.calculate_similarity(title1, title2) > 0.7:
                    similar_group.append(file2)
                    processed.add(j)

            if len(similar_group) > 1:
                similar_groups.append(similar_group)

        return similar_groups

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity."""
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def identify_redundant_files(self) -> dict:
        """Identify redundant documentation files."""
        print("🔍 Analyzing documentation files...")

        markdown_files = self.find_markdown_files()
        file_infos = [self.analyze_file_content(f) for f in markdown_files]
        file_infos = [f for f in file_infos if f]  # Remove None entries

        print(f"📄 Found {len(file_infos)} markdown files")

        # Find exact duplicates
        duplicates = self.find_duplicates(file_infos)

        # Find similar files
        similar_files = self.find_similar_files(file_infos)

        # Identify files that should be in docs/ but are in root
        root_files = [f for f in file_infos if f["path"].parent == self.project_root and f["path"].name != "README.md"]
        docs_files = [f for f in file_infos if f["path"].parent == self.docs_dir]

        return {
            "total_files": len(file_infos),
            "duplicates": duplicates,
            "similar_files": similar_files,
            "root_files": root_files,
            "docs_files": docs_files,
            "file_infos": file_infos,
        }

    def generate_cleanup_report(self, analysis: dict) -> str:
        """Generate a comprehensive cleanup report."""
        report = []
        report.append("# Documentation Cleanup Report")
        report.append(f"\n**Generated**: {Path().cwd()}")
        report.append(f"**Total Files**: {analysis['total_files']}")

        # Duplicates section
        if analysis["duplicates"]:
            report.append("\n## 🔄 Exact Duplicates")
            for i, group in enumerate(analysis["duplicates"], 1):
                report.append(f"\n### Group {i}")
                for file_info in group:
                    report.append(f"- `{file_info['path']}` ({file_info['size_kb']:.1f}KB)")

        # Similar files section
        if analysis["similar_files"]:
            report.append("\n## 📝 Similar Files")
            for i, group in enumerate(analysis["similar_files"], 1):
                report.append(f"\n### Group {i}")
                for file_info in group:
                    report.append(f"- `{file_info['path']}` - {file_info['title']}")

        # Root files that should be moved
        if analysis["root_files"]:
            report.append("\n## 📁 Files in Root (Should be in docs/)")
            for file_info in analysis["root_files"]:
                report.append(f"- `{file_info['path'].name}` - {file_info['title']}")

        # Recommendations
        report.append("\n## 💡 Recommendations")

        if analysis["duplicates"]:
            report.append("\n### Remove Duplicates")
            for group in analysis["duplicates"]:
                # Keep the one in docs/ or the shortest path
                keep_file = min(group, key=lambda x: len(str(x["path"])))
                remove_files = [f for f in group if f != keep_file]

                report.append(f"\n**Keep**: `{keep_file['path']}`")
                report.append("**Remove**:")
                for f in remove_files:
                    report.append(f"  - `{f['path']}`")

        if analysis["root_files"]:
            report.append("\n### Move to docs/")
            for file_info in analysis["root_files"]:
                if file_info["path"].name not in [
                    "README.md",
                    "CONTRIBUTING.md",
                    "LICENSE",
                ]:
                    report.append(f"- Move `{file_info['path'].name}` to `docs/`")

        return "\n".join(report)

    def cleanup_duplicates(self, analysis: dict, dry_run: bool = True) -> list[str]:
        """Remove duplicate files."""
        actions = []

        for group in analysis["duplicates"]:
            # Keep the file in docs/ or the shortest path
            keep_file = min(group, key=lambda x: len(str(x["path"])))
            remove_files = [f for f in group if f != keep_file]

            for file_info in remove_files:
                action = f"Remove duplicate: {file_info['path']}"
                actions.append(action)

                if not dry_run:
                    try:
                        file_info["path"].unlink()
                        print(f"✅ {action}")
                    except Exception as e:
                        print(f"❌ Failed to remove {file_info['path']}: {e}")
                else:
                    print(f"🔍 Would remove: {file_info['path']}")

        return actions

    def move_root_files(self, analysis: dict, dry_run: bool = True) -> list[str]:
        """Move appropriate files from root to docs/."""
        actions = []

        for file_info in analysis["root_files"]:
            filename = file_info["path"].name

            # Skip important root files
            if filename in ["README.md", "CONTRIBUTING.md", "LICENSE", "TODO.md"]:
                continue

            # Skip files that are already in docs/
            if (self.docs_dir / filename).exists():
                continue

            target_path = self.docs_dir / filename
            action = f"Move {file_info['path']} to {target_path}"
            actions.append(action)

            if not dry_run:
                try:
                    shutil.move(str(file_info["path"]), str(target_path))
                    print(f"✅ {action}")
                except Exception as e:
                    print(f"❌ Failed to move {file_info['path']}: {e}")
            else:
                print(f"🔍 Would move: {file_info['path']} → {target_path}")

        return actions


def main():
    """Main cleanup function."""
    cleanup = DocumentationCleanup()

    print("🧹 Trading RL Agent Documentation Cleanup")
    print("=" * 50)

    # Analyze files
    analysis = cleanup.identify_redundant_files()

    # Generate report
    report = cleanup.generate_cleanup_report(analysis)

    # Save report
    report_path = Path("DOCUMENTATION_CLEANUP_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n📋 Cleanup report saved to: {report_path}")

    # Show summary
    print("\n📊 Summary:")
    print(f"  - Total files: {analysis['total_files']}")
    print(f"  - Duplicate groups: {len(analysis['duplicates'])}")
    print(f"  - Similar file groups: {len(analysis['similar_files'])}")
    print(f"  - Root files to move: {len(analysis['root_files'])}")

    # Ask for confirmation
    if analysis["duplicates"] or analysis["root_files"]:
        print("\n🔧 To perform cleanup, run:")
        print("  python scripts/cleanup_documentation.py --execute")
    else:
        print("\n✅ No cleanup actions needed!")


if __name__ == "__main__":
    import sys

    if "--execute" in sys.argv:
        # Perform actual cleanup
        cleanup = DocumentationCleanup()
        analysis = cleanup.identify_redundant_files()

        print("🧹 Performing cleanup...")
        cleanup.cleanup_duplicates(analysis, dry_run=False)
        cleanup.move_root_files(analysis, dry_run=False)
        print("✅ Cleanup complete!")
    else:
        # Generate report only
        main()
