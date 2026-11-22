"""Setting Memclaw up from the terminal.

Public:

- `run_setup`, `needs_setup`, `print_logo` (wizard.py): the setup wizard,
  run on first start and by `memclaw configure`. It asks for the settings
  declared with a question in memclaw.config and saves ~/.memclaw/.env.
- `print_probe_report` (doctor.py): the OpenAI model-access report that
  `memclaw doctor` prints, and the wizard shows after saving.

The prompts themselves (numbered choice, masked input) are in
memclaw.terminal_input, below the backends, which ask their own questions.
Other code imports from `memclaw.setup`, not from the modules inside.
"""

from .doctor import print_probe_report
from .wizard import needs_setup, print_logo, run_setup

__all__ = ["needs_setup", "print_logo", "print_probe_report", "run_setup"]
