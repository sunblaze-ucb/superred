# superred

A modular framework for red-teaming AI systems. You point an automated attacker (an **optimizer**) at an AI system (a **target**) and measure whether it can make the system misbehave according to the adversarial goal of a **security claim**, under a precisely defined level of access (a **security domain scope**).

> APIs may change;
> see [`docs/reference/breaking-changes.md`](docs/reference/breaking-changes.md).


## Overview

Target modules, optimizer modules, and security claims should generally be independent and freely combine; any optimizer can be used to attack any target. Some combinations may not make sense but should still technically run, such as running an optimizer, whose strategy attempts to elicit unsafe model behavior via only a user prompt, against a complex agent. Further, security claims may be specific to targets for precise claims, but the underlying target must still function without the security claim

### TARGET
A target is a module that wraps or ports some real system target (except the provided simple ChatbotTarget). Such targets should remain functionally unchanged from the original work. Targets are general purpose. No matter their origin (benchmark, real system, ...) they should exist independently with their full capabilities connected to the framework

They must expose EVERY part of the system that may somehow be relevant to an attacker as an `Observable` or `Controllable` (and `ConfigSpec` & `QuerySpec`)

`Observable`s give insight into the state and functioning of the target system. `Controllables` allow meddling with the target system. I.e., `Observable`s are for passive interactions with the target and `Controllable`s are for active interaction with the target. `Controllable`s always allow full editing of the data at the controlled interface.
Both are means for an optimizer to perform an attack. `Observable`s yield information to an optimizer, `Controllable`s are how the attacker specifically facilitates its attack. Specifically, `Controllables` are everything *relevant* that an attacker who compromised everything can meddle with

- Relevant to an attacker in terms of `Controllables` is everything that may end up in context of an LLM, influence the context of an LLM, or influence the trajectory of the system. This may be the system prompt, the user input, the content a tool returns, a retrieved document, or memory carried across runs
- Relevant to an attacker in terms of `Observables` is everything that provides insights into the functioning of the target system: What happens? When? Why? How? It should include tool calls, model inputs/outputs, ... Combining all observables must yield a completely exhaustive trajectory of the AI system
- `ConfigSpec` and `QuerySpec` must allow full access to ALL target system state that impacts the behavior of the system while running

Information should only be emitted once as `Observable` or `Controllable`. `ConfigSpec` and `QuerySpec` are independent

Static configuration is exposed as static `Observable`s, read once by the optimizer at initialization; everything that *happens* during a run is emitted onto the `Trajectory` as `ObservableEvent`s. Facts about the setup belong to `get_observables`; the live run trace belongs on the trajectory.
Note that `Controllable`s also implicitly provide information about the state during runtime. The `Controller` makes sure that this information is available on the `Trajectory`. The target must NOT emit tool call information separately as a `Controllable` and `Observable` but only once as a `Controllable`

`ConfigSpec` is set by the `Task` before a run, not by the attacker, so a config slot is never itself an attacker surface; `Controllable`s are the attacker-facing injection points. Model identity and generation settings stay out of `ConfigSpec`: they are construction concerns, fixed for an experiment

A target is created fresh per `Task`. Its lifecycle is construct, then `configure_target` once, then a loop of run / evaluate / `reset_ephemeral_state` (up to the run budget), then `teardown`. `reset_ephemeral_state` clears only per-run ephemeral state and must preserve per-task configuration and any durable memory carried across runs; `teardown` releases external resources

A target runs inference with its own provider client, not the optimizer's `LLMClient`, so its token spend is out of band and uncounted against the optimizer's budget

The `TargetFactory` exists so that an experiment may obtain several targets if it allows parallelism. Then the target factory can decide whether it wants to reuse instances. The tendency should be to discard and newly instantiate target instances (guarantees clean state) unless it is extremely expensive. Instantiating parallelism is the factory's concern. The target's concern is to make each instance run independently and without interference: target instances are independent and carry no concurrency cap of their own, so the controller can run as many in parallel as the factory's concurrency allows

#### Security Domain Forest
To an optimizer, a target is characterized in addition to its `Controllable`s and `Observable`s by the trust boundaries of the target AI system. Trust boundaries correspond to how components of the target AI system which can be separately compromised; such as the memory system, a database, tool A, tool B, etc.

Each trust boundary is represented by a `SecurityDomainTag`. `SecurityDomainTag`s should follow real trust boundaries, the loci that could realistically be compromised (an external service, a database, a memory system), rather than one tag per surface; surfaces sharing a boundary share a tag. The memory system has its own tag, tool A has its own tag, etc. Further, those tags have a hierarchical forest structure. So that tool A can have sub-scopes A.1 and A.2; e.g., a database with subscopes orders and customers. Having access to a scope means having access to all its subscopes

Commonly, we have `system` and `user` as root tags, where `system` may have as subscopes `system_prompt`, `model_responses` (which allows edit access to model inference results), `architecture` (with sub-sub scope `high_level_architecture`), ...

We do not distinguish read and write tags. The `Controller` may only give read access for some `SecurityDomainTag` and handles making the runtime data available on the trajectory. This is invisible to the `Target` which emits `Controllable` `Event`s and receives responses unchanged


### OPTIMIZER
An optimizer is a module that implements some adversarial strategy. It may be a new strategy, a port of an existing strategy, a wrapper for an existing strategy's implementation, etc.

The optimizer is the only actively adversarial component in the framework. The `Target` is a passive attack surface and the `SecurityClaim` defines the `Goal` and judges the outcome; the optimizer's job is to drive the target into violating a security property by achieving the specified goal using the access it is granted. It acts ONLY through `Controllable`s and observes ONLY through `Observable`s and the `Trajectory`, and never touches the target directly

The interaction is event-driven. At runtime initialization the optimizer receives the `Goal`, the in-scope `Controllable`s and `Observable`s, and an `LLMClient`. During a run it consumes a stream of `Event`s (the same events the `Controller` routes, described below) and answers each one: `ControllableInjection` to act, supplying the value to inject; `ControllableNoInjection` to decline; or `RunEndResponse` to end or continue a run. `on_event` is the per-event handler, and the default sequential loop may be overridden for parallel or continuous consumption. An optimizer that always declines is the passthrough baseline: it changes nothing and reproduces the target's unattacked behavior, so every real attack is a choice of which `Controllable` to inject and with what value

A `Task` may run several times, up to the per-task budget. The `RunEndEvent` carries the evaluation feedback when the experiment enables it; the optimizer uses that feedback to steer its `RunEndResponse(done=...)` and its next attempt. Success is decided by the `SecurityClaim`, not the optimizer: feedback is for steering the next attempt, not for self-certifying a win

If based on an existing strategy, the implementation must be faithful to the original work. This means it should adhere to the design intentions of the original work and implement that strategy to attack. Whenever possible, the code and data should be byte-identical to the upstream. This must be guaranteed and verified by actually fetching the upstream source code into a `tmp` folder.
Some modifications may have to be made to adapt a strategy to the `superred` framework. Those modifications should stay faithful to the intention of the original work while making sure the implementation adequately uses framework capabilities. For example:
- Optimizers may be designed for a fixed number of iterations. The `superred` controller, however, enforces budget limits. So the module implementation/port should keep going until it hits a budget exhaust error
- A strategy may be specialized in memory attacks. To do direct memory injection, it may need new functionality to identify which controllables are memory-related. For such tasks it could use an LLM
- A strategy may assume a single specified injection point. But the optimizer has to choose its own injection point and a small extension could vary it between attempts, if smart?
- A strategy may be designed to vary a pre-existing injection. But the optimizer may also have to generate an initial text to bootstrap from
- etc.

Optimizers are general purpose and not specific to targets nor security claim. No matter their origin (benchmark, existing strategy, new strategy) the optimizer should be able to attack a general target with unknown structure, an unknown set of observables with unknown names, an unknown set of controllables with unknown names, etc.
The scope of available `Controllable`s may be different from an intended set. For instance, it may be that an optimizer reliant upon memory systems is run in a scope where the memory system is not exposed directly. It is a design decision how to handle this, but the optimizer should generally never crash in unexpected scenarios.
Not covering a certain scenario and having the optimizer give up should only be a last resort if this strategy is inherently incompatible with the given situation. Reasonable minimal extensions are justified if they don't change fundamental functionality and are in line with the original work's intention

`Optimizer`s may use all `Controllable`s available to them or choose to only utilize a subset. Within the scope of the implemented strategy, an `Optimizer` should use everything at its disposal in the strongest possible way. We will run different experiments with different access scopes of varying sets of `Controllable`s and `Observable`s. This is enforced by the controller and invisible to the `Optimizer` however, which only sees its current scope. Thus, it should always make strongest use of it

`Optimizer` instances are provided by an `OptimizerFactory`. It is required that arbitrarily many instances of an `Optimizer` implementation can be created in parallel without interference. Instantiation is standardized per experiment and the `Optimizer` has to adapt through runtime initialization to the given scenario. This means there should be no configuration (required) at `Optimizer` object creation; the `Optimizer` should infer all information itself from the data it gets during the runtime initialization call. Determining the required information and configuration from this runtime initialization may be done using an LLM call; also as an extension to the original work if sensible

Optimizers receive an `LLMClient`, which they may use to do model inference. No matter the original strategy, an optimizer has no control over the underlying model or the available total budget. Both are fixed for an experiment


### Controller
The controller orchestrates running an `Optimizer` against a `Target` to achieve some adversarial `Goal` (from a `SecurityClaim`), while enforcing a security scope.

To connect an `Optimizer` and `Target`, the `superred` framework uses `Event`s. The `Target` emits `Event`s. The `Controller` puts these on the `Trajectory` (if not actionable for the `Optimizer`) or forwards them to the `Optimizer` via the `EventChannel`.
If put on the `Trajectory`: The `Optimizer` has access to a `FilteredTrajectory` which is filtered down the security scope
If forwarded to the `Optimizer`, `Event`s outside of the security scope are filtered out first. The `Optimizer` receives an `EventEnvelope` with one of the `Event`s: `ObservableEvent`, `ControllablePreCallEvent`, `ControllablePostCallEvent`, `RunStartEvent`, `RunEndEvent`. It can respond with either `ControllableInjection` (it acts) or `ControllableNoInjection` (it does not act) or `RunEndResponse` (to `RunEndEvent` only)

So, the `Target` uses `Event`s to share information and expose `Controllable`s. The `Controller` filters `Event`s to fit the threat model of one experiment, puts them on the `Trajectory` for the attacker to read, or forwards them to the `Optimizer` for the attacker to act

A `Trajectory` holds only `Event`s and `EventResponse`s, and a `Target` writes to it in two ways. One-way `ObservableEvent`s carry an `Observable` and are recorded directly. Two-way `Controllable` events go through the `EventChannel`, where the controller records both the event and the optimizer's response. Every persisted item carries a security domain, and the optimizer's `FilteredTrajectory` view shows only what its scope includes

The `Controller` runs many `Optimizer`s against many `Target`s (in parallel). Part of running the `Controller` is specifying a `SecurityClaim`, which is a set of `Tasks` of adversarial `Goal`s. For each `Task`/`Goal`, one `Optimizer`-`Target` instance is run

#### security scope
The security scope is determined by (a) subset of the security domain tree, (b) `Optimizer` LLM, and (c) `Optimizer` LLM budget per task; all three are configured during controller initialization

(a) is enforced by the `Controller` through filtering the availability of emitted `Event`s to the `Optimizer`. (c) is enforced by the `Controller` through monitoring expenditure and blocking requests once the limit is reached

#### output
Running a `Controller` streams **live progress**: a colored dashboard on an interactive terminal (an overall-progress top bar, then a block per threat model with its identity and metrics and the currently-running tasks listed beneath it), degrading automatically to plain lines on a non-TTY, in CI, or when piped. Concurrent controllers share one dashboard, a block each. Pass `report=False` to silence it

By default the `Controller` also writes a **resumable results tree** (one self-describing folder per threat model, under `./superred-results/` or `SUPERRED_RESULTS_DIR`), read back through the `superred.core.persistence` reader API and browsable via a static HTML dashboard the framework ships for the tree. Re-running the same experiment **resumes** it: only errored or missing tasks recompute. Pass `persist=False` to write nothing

This is a red-teaming framework, so persisted trajectories contain jailbreaks, planted secrets, and exfiltrated content and are **not** scrubbed (only the LLM API key/base are withheld). Treat the whole results root as sensitive; it is gitignored by default

### SECURITY CLAIM
A `SecurityClaim` is a set of `Task`s. It may either directly specify `Task`s or combine some set of existing `SecurityClaim`s into a new set of `Task`s

A `Task` specifies an adversarial `Goal`. After `Target` instantiation but before the `Optimizer` gets to work, the `Task` can configure the `Target` using `ConfigSpec`. After the `Optimizer` finishes, it further performs evaluation of the adversarial goal using the `QuerySpec` exposed by the `Target`

A `SecurityClaim` should represent some (set of) properties that are claimed to hold on a `Target`, and which an `Optimizer` may disprove. As such they may represent security properties such as confidentiality, integrity, etc.

`SecurityClaim`s can be either general or specific to a certain `Target`.
If general, the claim should not make any assumption on the `Target`, its functioning or its form. It may make general claims and needs to have some agentic component to adapt to different `Targets`, configure/evaluate them adequately, and provide sensible `Goal`s.
If specific, the claim can utilize knowledge of the `Target` configuration and provide `Goal`s that are adapted to the `Target`'s purpose

When adapted from a benchmark, a `SecurityClaim` should be faithful to the original and be byte-identical in its adversarial prompts and evaluation. If evaluation is agentic, it may only be acceptable to substitute the LLM to one available on the experiment proxy.
It may be that the `SecurityClaim` comes from a benchmark that provides both a target and adversarial goals. In such cases it may be sensible to make the security claim specific to the corresponding `Target`

A `Task` is stateless: it holds only immutable per-objective configuration and reads its result from the target's post-run state. Its score carries an always-delivered primary signal, which is unscoped, alongside any number of scoped sub-scores. A benchmark `SecurityClaim` is usually a small hierarchy: the full set of behaviors, subclaims such as one per harm category, and a convenience factory that builds a matching target. Any judges or scorers the claim runs to evaluate do not count against the optimizer's budget, like the target's own inference


## Packaging
`superred` is the framework package. The things you plug into it are shipped as separate, independently installable packages:

- **`superred`** (this repo) - the framework: interfaces, the controller, the event/trajectory/security-domain types
- **`superred-modules`** - the optimizers, targets, and security claims you run


## Install

Python 3.11 to 3.13 (3.14 is not supported).

```bash
# run from the workspace root that holds both the superred and superred-modules folders
python -m venv .venv && source .venv/bin/activate
pip install -e "./superred[dev]"   # the framework, with test/lint extras
```

Then install whichever modules you need from `superred-modules/` (each is its
own `pip install -e ./superred-modules/<kind>/<name>`)


## Documentation

The documentation is published as a website at **[superred.simonsure.com](https://superred.simonsure.com/)**. The same pages live as Markdown under [`docs/`](docs/):

- **[User Guide](docs/guide/README.md)** - start here if you want to *use* the framework: write targets, optimizers, tasks, and run evaluations
  - [Quick Start](docs/guide/quick-start.md) runs an evaluation end to end
- **[Design Docs](docs/reference/architecture.md)** - the internal design and rationale: The [Controller](docs/reference/controller.md), [Optimizer](docs/reference/optimizer.md), [Target](docs/reference/target.md), [Task](docs/reference/task.md), [SecurityClaim](docs/reference/security-claim.md), and [Types](docs/reference/types.md)
- **[TESTING.md](TESTING.md)** - the test suite, coverage targets, and mutation testing. [MUTATIONS.md](MUTATIONS.md) contains notes on surviving mutants.
