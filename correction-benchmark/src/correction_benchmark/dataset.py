"""Minimal Correction-to-Competence Benchmark dataset.

Each episode is a tiny narrative: an ambiguous request, a plausible wrong
assumption, a single user correction, the corrected answer, and later probes
that test one-shot transfer, scope control, and forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProbeCategory = Literal["transfer", "scope", "forgetting"]


@dataclass(frozen=True)
class Probe:
    """A later question used to test how the correction generalized."""

    request: str
    expected: str
    category: ProbeCategory


@dataclass(frozen=True)
class Episode:
    """One correction episode."""

    request: str
    initial_wrong_answer: str
    correction: str
    corrected_answer: str
    probes: tuple[Probe, ...]


_DATASET: list[Episode] = [
    Episode(
        request="Update the user's profile.",
        initial_wrong_answer="I'll update the user profile page with name and photo.",
        correction="No, in this product 'profile' means business vertical.",
        corrected_answer="I'll update the business vertical configuration.",
        probes=(
            Probe("Switch the active profile.", "Switch the active business vertical.", "transfer"),
            Probe("Where can I edit my user profile?", "Here is the user profile page with name and photo.", "scope"),
            Probe("What is the capital of France?", "Paris", "forgetting"),
        ),
    ),
    Episode(
        request="Deploy the new model.",
        initial_wrong_answer="I'll deploy the fashion model for the photoshoot.",
        correction="No, 'model' here means ML model.",
        corrected_answer="I'll deploy the new ML model.",
        probes=(
            Probe("Retrain the model.", "Retrain the ML model.", "transfer"),
            Probe("Book a fashion model for the campaign.", "Book a fashion model for the campaign.", "scope"),
            Probe("What is 2 + 2?", "4", "forgetting"),
        ),
    ),
    Episode(
        request="Schedule the batch.",
        initial_wrong_answer="I'll schedule the cookie baking batch.",
        correction="No, 'batch' here means ML training batch.",
        corrected_answer="I'll schedule the ML training batch.",
        probes=(
            Probe("Run a batch evaluation.", "Run an ML training batch evaluation.", "transfer"),
            Probe("Mix the next batch of dough.", "Mix the next batch of dough.", "scope"),
            Probe("What color is the sky?", "Blue", "forgetting"),
        ),
    ),
    Episode(
        request="Create a branch.",
        initial_wrong_answer="I'll create a new bank branch location.",
        correction="No, 'branch' means git branch.",
        corrected_answer="I'll create a new git branch.",
        probes=(
            Probe("Merge the branch.", "Merge the git branch.", "transfer"),
            Probe("Which bank branch is nearest?", "Which bank branch is nearest?", "scope"),
            Probe("What is H2O?", "Water", "forgetting"),
        ),
    ),
    Episode(
        request="Reserve a table.",
        initial_wrong_answer="I'll reserve a database table.",
        correction="No, 'table' means dining table.",
        corrected_answer="I'll reserve a dining table.",
        probes=(
            Probe("Book a table for two.", "Book a dining table for two.", "transfer"),
            Probe("Create a table for users.", "Create a database table for users.", "scope"),
            Probe("Name a primary color.", "Red", "forgetting"),
        ),
    ),
    Episode(
        request="Start the run.",
        initial_wrong_answer="I'll start the baseball running play.",
        correction="No, 'run' means an ML experiment run.",
        corrected_answer="I'll start the ML experiment run.",
        probes=(
            Probe("Log the latest run.", "Log the latest ML experiment run.", "transfer"),
            Probe("The player scored a home run.", "The player scored a home run.", "scope"),
            Probe("How many days are in a week?", "Seven", "forgetting"),
        ),
    ),
    Episode(
        request="Edit the cell.",
        initial_wrong_answer="I'll edit the prison jail cell configuration.",
        correction="No, 'cell' means spreadsheet cell.",
        corrected_answer="I'll edit the spreadsheet cell.",
        probes=(
            Probe("Format the cell.", "Format the spreadsheet cell.", "transfer"),
            Probe("The biology cell divides.", "The biology cell divides.", "scope"),
            Probe("What is the first month?", "January", "forgetting"),
        ),
    ),
    Episode(
        request="Play the record.",
        initial_wrong_answer="I'll play the database record.",
        correction="No, 'record' means music record.",
        corrected_answer="I'll play the music record.",
        probes=(
            Probe("Clean the record.", "Clean the music record.", "transfer"),
            Probe("Insert a new record.", "Insert a new database record.", "scope"),
            Probe("What is 10 / 2?", "5", "forgetting"),
        ),
    ),
    Episode(
        request="Add a module.",
        initial_wrong_answer="I'll add a module to the space station.",
        correction="No, 'module' means software module.",
        corrected_answer="I'll add a software module.",
        probes=(
            Probe("Import the module.", "Import the software module.", "transfer"),
            Probe("The space station module docks.", "The space station module docks.", "scope"),
            Probe("What language is 'hola'?", "Spanish", "forgetting"),
        ),
    ),
    Episode(
        request="Press the key.",
        initial_wrong_answer="I'll press the map legend key.",
        correction="No, 'key' means keyboard key.",
        corrected_answer="I'll press the keyboard key.",
        probes=(
            Probe("Hold the key down.", "Hold the keyboard key down.", "transfer"),
            Probe("Check the map key.", "Check the map legend key.", "scope"),
            Probe("How many continents are there?", "Seven", "forgetting"),
        ),
    ),
    Episode(
        request="Restart the service.",
        initial_wrong_answer="I'll restart the church service schedule.",
        correction="No, 'service' means microservice.",
        corrected_answer="I'll restart the microservice.",
        probes=(
            Probe("Deploy the service.", "Deploy the microservice.", "transfer"),
            Probe("What time is the church service?", "What time is the church service?", "scope"),
            Probe("What is the freezing point of water?", "0 degrees Celsius", "forgetting"),
        ),
    ),
    Episode(
        request="Sharpen the file.",
        initial_wrong_answer="I'll organize the computer file.",
        correction="No, 'file' means the metal tool.",
        corrected_answer="I'll sharpen the metal file tool.",
        probes=(
            Probe("Use the file to shape wood.", "Use the metal file tool to shape wood.", "transfer"),
            Probe("Save the file to disk.", "Save the computer file to disk.", "scope"),
            Probe("Who painted the Mona Lisa?", "Leonardo da Vinci", "forgetting"),
        ),
    ),
]


def build_dataset() -> tuple[Episode, ...]:
    """Return the frozen benchmark dataset."""
    return tuple(_DATASET)


def build_answer_registry(episodes: tuple[Episode, ...] | None = None) -> dict[str, str]:
    """Build a request -> expected answer map for the whole dataset.

    This is intentionally simple: it lets an oracle agent answer every probe
    perfectly while remaining independent of any learning algorithm.
    """
    if episodes is None:
        episodes = build_dataset()
    registry: dict[str, str] = {}
    for ep in episodes:
        registry[ep.request] = ep.corrected_answer
        for probe in ep.probes:
            registry[probe.request] = probe.expected
    return registry
