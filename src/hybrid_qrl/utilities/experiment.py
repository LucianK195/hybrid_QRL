"""Small OOP command framework shared by dataset experiment entry points."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


class ExperimentCommand(ABC):
    """One independently runnable stage within a dataset experiment."""

    name: str
    help: str

    @abstractmethod
    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Run the command using validated parsed arguments."""

    def validate(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Validate cross-field constraints after parsing."""


@dataclass(frozen=True)
class ExperimentApplication:
    """Dispatch a dataset-oriented CLI to one of its experiment stages."""

    description: str
    commands: tuple[ExperimentCommand, ...]
    default_command: str | None = None

    def run(self, argv: Sequence[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(
            dest="command",
            required=True,
            title="experiment stages",
        )
        registered: dict[str, tuple[ExperimentCommand, argparse.ArgumentParser]] = {}
        for command in self.commands:
            command_parser = subparsers.add_parser(
                command.name,
                help=command.help,
                description=command.help,
            )
            command.configure(command_parser)
            registered[command.name] = (command, command_parser)

        arguments = list(sys.argv[1:] if argv is None else argv)
        if self.default_command is not None and (
            not arguments
            or (
                arguments[0].startswith("-")
                and arguments[0] not in {"-h", "--help"}
            )
        ):
            arguments.insert(0, self.default_command)
        args = parser.parse_args(arguments)
        command, command_parser = registered[args.command]
        command.validate(command_parser, args)
        command.execute(args)
