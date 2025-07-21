#!/usr/bin/env python3
"""
Documentation Standardization Script

This script renames documentation files according to the standardized
naming convention: [category]_[feature]_[type].md
"""

import re
from pathlib import Path


class DocumentationStandardizer:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.rename_mapping = {}

    def get_standardized_name(self, original_name: str, title: str = "") -> str:
        """Convert a filename to the standardized naming convention."""

        # Remove .md extension
        base_name = original_name.replace(".md", "")

        # Define category mappings based on filename patterns and content
        category_mappings = {
            # User guides
            "README_CLI_USAGE": "user-guide_cli_reference",
            "DEMO_README": "user-guide_demo_guide",
            "demo_commands": "user-guide_demo_commands",
            "getting_started": "user-guide_getting-started_quickstart",
            # Developer guides
            "DEVELOPMENT_GUIDE": "developer_setup_guide",
            "TESTING_GUIDE": "developer_testing_guide",
            "EVALUATION_GUIDE": "developer_evaluation_guide",
            # Feature documentation
            "enhanced_training_guide": "feature_cnn-lstm-training_guide",
            "ADVANCED_POLICY_OPTIMIZATION": "feature_reinforcement-learning_guide",
            "ENSEMBLE_SYSTEM_GUIDE": "feature_ensemble-system_guide",
            "RISK_ALERT_SYSTEM": "feature_risk-management_guide",
            "transaction_cost_modeling": "feature_transaction-costs_guide",
            "backtest_evaluator": "feature_backtesting_guide",
            "scenario_evaluation": "feature_scenario-evaluation_guide",
            "unified_config_schema": "feature_configuration_guide",
            # Integration & deployment
            "ALPACA_INTEGRATION": "integration_alpaca_guide",
            "ALPACA_INTEGRATION_SUMMARY": "integration_alpaca_summary",
            "CI_CD_PIPELINE_DOCUMENTATION": "deployment_cicd_guide",
            "PRODUCTION_IMPLEMENTATION_PLAN": "deployment_production_guide",
            "SECURITY_COMPLIANCE_FRAMEWORK": "deployment_security_guide",
            # Development & operations
            "DEPENDENCY_MANAGEMENT": "developer_dependencies_guide",
            "ENV_VARIABLES_INTEGRATION_SUMMARY": "developer_environment_guide",
            "PERFORMANCE_TESTING_FRAMEWORK": "developer_performance_guide",
            "SYSTEM_HEALTH_MONITORING": "operations_monitoring_guide",
            # Project management
            "PROJECT_STATUS": "project_status_report",
            "IMPLEMENTATION_ROADMAP_DETAILED": "project_roadmap_guide",
            "PRODUCTION_READINESS_ASSESSMENT": "project_readiness_report",
            "PRODUCTION_TRAJECTORY_ROADMAP": "project_trajectory_guide",
            # Quality assurance
            "QA_VALIDATION_REPORT": "qa_validation_report",
            "TEST_SUITE_OPTIMIZATION_SUMMARY": "qa_testing_optimization",
            "DATA_PIPELINE_TEST_COVERAGE_REPORT": "qa_data-pipeline_coverage",
            "CLI_COVERAGE_IMPROVEMENT_SUMMARY": "qa_cli_coverage",
            # Architecture & design
            "DOCUMENTATION_STANDARDS": "architecture_documentation_standards",
            "KNOWLEDGE_MANAGEMENT_FRAMEWORK": "architecture_knowledge_management",
            "SPRINT_PLANNING_FRAMEWORK": "architecture_sprint_planning",
            "TASK_TRACKING_AND_PROGRESS_MONITORING": "architecture_task_tracking",
            # Legacy and cleanup files
            "DOCUMENTATION_ACTION_PLAN": "legacy_documentation_action",
            "DOCUMENTATION_AUDIT_REPORT": "legacy_documentation_audit",
            "DOCUMENTATION_INVENTORY": "legacy_documentation_inventory",
            "DOCUMENTATION_UPDATE_SUMMARY": "legacy_documentation_update",
            "DOCUMENTATION_CLEANUP_REPORT": "legacy_cleanup_report",
            "CLEANUP_SUMMARY": "legacy_cleanup_summary",
            "DEMO_SETUP_SUMMARY": "legacy_demo_setup",
            "TRANSACTION_COST_MODELING_IMPLEMENTATION": "legacy_transaction_costs",
            # Examples and templates
            "examples": "examples_code_examples",
            "index": "index_main_index",
            "README": "index_docs_readme",
        }

        # Check if we have a direct mapping
        if base_name in category_mappings:
            return category_mappings[base_name] + ".md"

        # Try to infer category from title or filename
        title_lower = title.lower()
        base_lower = base_name.lower()

        # Infer category based on content
        if any(word in title_lower for word in ["guide", "tutorial", "how-to"]):
            if any(word in base_lower for word in ["user", "demo", "cli"]):
                return f"user-guide_{base_name.lower().replace('_', '-')}_guide.md"
            if any(word in base_lower for word in ["dev", "development", "test"]):
                return f"developer_{base_name.lower().replace('_', '-')}_guide.md"
            return f"feature_{base_name.lower().replace('_', '-')}_guide.md"

        if any(word in title_lower for word in ["report", "summary", "assessment"]):
            if any(word in base_lower for word in ["qa", "test", "coverage"]):
                return f"qa_{base_name.lower().replace('_', '-')}_report.md"
            if any(word in base_lower for word in ["project", "status", "roadmap"]):
                return f"project_{base_name.lower().replace('_', '-')}_report.md"
            return f"legacy_{base_name.lower().replace('_', '-')}_report.md"

        if any(word in title_lower for word in ["integration", "deployment", "production"]):
            return f"deployment_{base_name.lower().replace('_', '-')}_guide.md"

        # Default to legacy category for unknown files
        return f"legacy_{base_name.lower().replace('_', '-')}_document.md"

    def analyze_files(self) -> dict:
        """Analyze all markdown files and create rename mapping."""
        print("🔍 Analyzing documentation files for standardization...")

        markdown_files = list(self.docs_dir.glob("*.md"))
        analysis = {
            "total_files": len(markdown_files),
            "rename_mapping": {},
            "skipped_files": [],
            "errors": [],
        }

        for file_path in markdown_files:
            try:
                # Read file to get title
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Extract title from first heading
                title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                title = title_match.group(1) if title_match else ""

                # Get standardized name
                new_name = self.get_standardized_name(file_path.name, title)

                # Check if rename is needed
                if new_name != file_path.name:
                    analysis["rename_mapping"][str(file_path)] = new_name
                else:
                    analysis["skipped_files"].append(str(file_path))

            except Exception as e:
                analysis["errors"].append(f"Error processing {file_path}: {e}")

        return analysis

    def generate_standardization_report(self, analysis: dict) -> str:
        """Generate a report of the standardization analysis."""
        report = []
        report.append("# Documentation Standardization Report")
        report.append(f"\n**Generated**: {Path().cwd()}")
        report.append(f"**Total Files**: {analysis['total_files']}")
        report.append(f"**Files to Rename**: {len(analysis['rename_mapping'])}")
        report.append(f"**Files Skipped**: {len(analysis['skipped_files'])}")

        if analysis["rename_mapping"]:
            report.append("\n## 📝 Files to Rename")
            for old_path, new_name in analysis["rename_mapping"].items():
                old_name = Path(old_path).name
                report.append(f"- `{old_name}` → `{new_name}`")

        if analysis["skipped_files"]:
            report.append("\n## ✅ Files Already Standardized")
            for file_path in analysis["skipped_files"]:
                report.append(f"- `{Path(file_path).name}`")

        if analysis["errors"]:
            report.append("\n## ❌ Errors")
            for error in analysis["errors"]:
                report.append(f"- {error}")

        # Show naming convention examples
        report.append("\n## 📋 Naming Convention Examples")
        report.append("\n### Categories")
        report.append("- `user-guide` - User-facing documentation")
        report.append("- `developer` - Developer documentation")
        report.append("- `feature` - Feature-specific guides")
        report.append("- `integration` - Integration guides")
        report.append("- `deployment` - Deployment and operations")
        report.append("- `project` - Project management")
        report.append("- `qa` - Quality assurance")
        report.append("- `legacy` - Legacy documentation")

        report.append("\n### Types")
        report.append("- `guide` - Comprehensive how-to guide")
        report.append("- `reference` - Reference documentation")
        report.append("- `quickstart` - Quick start guide")
        report.append("- `report` - Reports and summaries")
        report.append("- `examples` - Code examples")

        return "\n".join(report)

    def execute_renames(self, analysis: dict, dry_run: bool = True) -> list[str]:
        """Execute the file renames."""
        actions = []

        for old_path, new_name in analysis["rename_mapping"].items():
            old_file = Path(old_path)
            new_file = self.docs_dir / new_name

            # Check for conflicts
            if new_file.exists() and not dry_run:
                print(f"⚠️  Warning: {new_file} already exists, skipping {old_file}")
                continue

            action = f"Rename {old_file.name} to {new_name}"
            actions.append(action)

            if not dry_run:
                try:
                    old_file.rename(new_file)
                    print(f"✅ {action}")
                except Exception as e:
                    print(f"❌ Failed to rename {old_file}: {e}")
            else:
                print(f"🔍 Would rename: {old_file.name} → {new_name}")

        return actions


def main():
    """Main standardization function."""
    standardizer = DocumentationStandardizer()

    print("📋 Trading RL Agent Documentation Standardization")
    print("=" * 60)

    # Analyze files
    analysis = standardizer.analyze_files()

    # Generate report
    report = standardizer.generate_standardization_report(analysis)

    # Save report
    report_path = Path("DOCUMENTATION_STANDARDIZATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n📋 Standardization report saved to: {report_path}")

    # Show summary
    print("\n📊 Summary:")
    print(f"  - Total files: {analysis['total_files']}")
    print(f"  - Files to rename: {len(analysis['rename_mapping'])}")
    print(f"  - Files already standardized: {len(analysis['skipped_files'])}")

    if analysis["rename_mapping"]:
        print("\n🔧 To perform standardization, run:")
        print("  python scripts/standardize_documentation.py --execute")
    else:
        print("\n✅ All files are already standardized!")


if __name__ == "__main__":
    import sys

    if "--execute" in sys.argv:
        # Perform actual standardization
        standardizer = DocumentationStandardizer()
        analysis = standardizer.analyze_files()

        print("📋 Performing standardization...")
        standardizer.execute_renames(analysis, dry_run=False)
        print("✅ Standardization complete!")
    else:
        # Generate report only
        main()
