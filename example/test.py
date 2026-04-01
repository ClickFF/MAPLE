#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAPLE Test Runner

A simple test runner for all .inp files in the example directory.
Runs each test case and collects results, similar to pytest output style.

Usage:
    python test.py              # Run all tests
    python test.py -v           # Verbose mode
    python test.py -k neb       # Run only tests matching 'neb'
    python test.py --list       # List all test cases without running
"""
import os
import sys
import time
import glob
import argparse
import traceback
from pathlib import Path

# Add parent directory to path for maple imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from maple.function.engine import engine


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.RED = cls.YELLOW = cls.CYAN = cls.BOLD = cls.END = ''


class TestResult:
    """Container for a single test result."""
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.passed = False
        self.skipped = False
        self.error = None
        self.duration = 0.0


class TestRunner:
    """Simple test runner for MAPLE input files."""

    def __init__(self, example_dir: str, verbose: bool = False, pattern: str = None):
        self.example_dir = example_dir
        self.verbose = verbose
        self.pattern = pattern.lower() if pattern else None
        self.results = []
        self.start_time = None

    def discover_tests(self) -> list:
        """Find all .inp files in example directory."""
        inp_files = []
        for root, dirs, files in os.walk(self.example_dir):
            for f in files:
                if f.endswith('.inp'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.example_dir)
                    # Create test name from relative path
                    test_name = rel_path.replace(os.sep, '::').replace('.inp', '')
                    inp_files.append((test_name, full_path))

        # Sort by path for consistent ordering
        inp_files.sort(key=lambda x: x[0])

        # Filter by pattern if specified
        if self.pattern:
            inp_files = [(n, p) for n, p in inp_files if self.pattern in n.lower()]

        return inp_files

    def run_single_test(self, test_name: str, inp_path: str) -> TestResult:
        """Run a single test case."""
        result = TestResult(test_name, inp_path)

        # Generate output path in same directory as input
        base_name = os.path.splitext(inp_path)[0]
        out_path = f"{base_name}.out"

        start = time.time()
        try:
            eng = engine()
            eng(inp_path, out_path)
            result.passed = True
        except Exception as e:
            result.passed = False
            result.error = str(e)
            if self.verbose:
                result.error += f"\n{traceback.format_exc()}"
        finally:
            result.duration = time.time() - start

        return result

    def print_test_status(self, result: TestResult, index: int, total: int):
        """Print status for a single test."""
        if result.passed:
            status = f"{Colors.GREEN}PASSED{Colors.END}"
            symbol = f"{Colors.GREEN}✓{Colors.END}"
        elif result.skipped:
            status = f"{Colors.YELLOW}SKIPPED{Colors.END}"
            symbol = f"{Colors.YELLOW}s{Colors.END}"
        else:
            status = f"{Colors.RED}FAILED{Colors.END}"
            symbol = f"{Colors.RED}✗{Colors.END}"

        duration = f"({result.duration:.2f}s)"
        print(f" {symbol} {result.name} {status} {duration}")

        if not result.passed and result.error and self.verbose:
            # Print error indented
            for line in result.error.split('\n'):
                print(f"   {Colors.RED}{line}{Colors.END}")

    def run(self):
        """Run all discovered tests."""
        tests = self.discover_tests()

        if not tests:
            print(f"{Colors.YELLOW}No tests found matching criteria{Colors.END}")
            return 0

        # Print header
        print(f"\n{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"{Colors.CYAN}MAPLE Test Runner{Colors.END}")
        print(f"{Colors.BOLD}{'=' * 70}{Colors.END}")
        print(f"Discovered {len(tests)} test(s)\n")

        self.start_time = time.time()
        passed = 0
        failed = 0
        skipped = 0

        for i, (test_name, test_path) in enumerate(tests, 1):
            result = self.run_single_test(test_name, test_path)
            self.results.append(result)
            self.print_test_status(result, i, len(tests))

            if result.passed:
                passed += 1
            elif result.skipped:
                skipped += 1
            else:
                failed += 1

        total_time = time.time() - self.start_time

        # Print summary
        self.print_summary(passed, failed, skipped, total_time)

        return failed

    def print_summary(self, passed: int, failed: int, skipped: int, total_time: float):
        """Print test summary."""
        print(f"\n{Colors.BOLD}{'=' * 70}{Colors.END}")

        # Collect failed tests
        failed_tests = [r for r in self.results if not r.passed and not r.skipped]

        if failed_tests:
            print(f"\n{Colors.RED}{Colors.BOLD}FAILURES:{Colors.END}")
            print(f"{'-' * 70}")
            for result in failed_tests:
                print(f"\n{Colors.RED}{result.name}{Colors.END}")
                print(f"  Path: {result.path}")
                if result.error:
                    print(f"  Error: {result.error.split(chr(10))[0]}")
            print(f"{'-' * 70}")

        # Summary line
        total = passed + failed + skipped
        if failed == 0:
            status_color = Colors.GREEN
            status_text = "ALL PASSED"
        else:
            status_color = Colors.RED
            status_text = "SOME FAILED"

        print(f"\n{status_color}{Colors.BOLD}{status_text}{Colors.END}")
        print(f"  {Colors.GREEN}{passed} passed{Colors.END}, "
              f"{Colors.RED}{failed} failed{Colors.END}, "
              f"{Colors.YELLOW}{skipped} skipped{Colors.END} "
              f"in {total_time:.2f}s")
        print(f"{Colors.BOLD}{'=' * 70}{Colors.END}\n")

    def list_tests(self):
        """List all discovered tests without running them."""
        tests = self.discover_tests()

        print(f"\n{Colors.BOLD}Discovered {len(tests)} test(s):{Colors.END}\n")
        for test_name, test_path in tests:
            print(f"  {Colors.CYAN}{test_name}{Colors.END}")
            if self.verbose:
                print(f"    -> {test_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='MAPLE Test Runner - Run all example input files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (show full error traces)')
    parser.add_argument('-k', '--keyword', type=str, default=None,
                        help='Only run tests matching keyword')
    parser.add_argument('--list', action='store_true',
                        help='List tests without running')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    # Get example directory (same directory as this script)
    example_dir = os.path.dirname(os.path.abspath(__file__))

    runner = TestRunner(
        example_dir=example_dir,
        verbose=args.verbose,
        pattern=args.keyword
    )

    if args.list:
        runner.list_tests()
        return 0

    failed_count = runner.run()
    return 1 if failed_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
