\# Reachy Thinking Detector



\*\*Author:\*\* Marjan Beheshti, CS5761 Human-Robot Interaction, Michigan Technological University



A Reachy Mini robot answers spoken questions with a local LLM, and a detector

watches the model while it thinks to decide whether it is \*\*making progress\*\*

toward an answer or \*\*overthinking / stuck\*\*. The goal is to give a human

supervisor an honest signal about when to wait for the robot and when to take over.



\## How it works



Four programs run at the same time and talk over WebSockets:



```

&#x20;person speaks

&#x20;     |

&#x20;     v

&#x20;experiment/experiment.py  --- question text --->  wizardOfOz/index.js  --->  wizard\_brain.py

&#x20;(robot: microphone,                               (message server and         (LLM + detector,

&#x20; speech recognition,      <--- reply text -----    web dashboard)      <---   via Ollama)

&#x20; speaker)

&#x20;     |

&#x20;     v

&#x20;robot speaks the reply

```



1\. \*\*Robot side\*\* (`experiment/experiment.py`) records speech from the robot's

&#x20;  microphone, turns it into text with faster-whisper, and sends it to the server.

&#x20;  It runs three concurrent tasks (listen, receive, speak) and ignores the

&#x20;  microphone while the robot is talking, so it never hears its own voice.

2\. \*\*Server\*\* (`wizardOfOz/index.js`) passes messages between the robot and the

&#x20;  brain, and shows them on a web dashboard where a person can also type replies

&#x20;  (the Wizard-of-Oz method).

3\. \*\*Brain\*\* (`wizard\_brain.py`) lets a reasoning model think about the question,

&#x20;  checks its answer so far every 25 tokens, and replies once the answer stops changing.

4\. \*\*Ollama\*\* runs the language models locally: no API key or GPU needed.



\## Project structure



```

wizard\_brain.py           LLM + detector that answers the robot's questions

watch\_messages.py         debugging tool: prints every message on the server

experiment/

&#x20; experiment.py           robot side of the conversation

&#x20; record.py               microphone recording and speech-to-text

&#x20; text\_to\_speech.py       speech output (Hugging Face or offline voice)

&#x20; robot.py                connection to the robot

wizardOfOz/

&#x20; index.js                message server

&#x20; index.html              web dashboard

entropy/                  research scripts for the detector (see below)

```



\## Setup



1\. Install \[Ollama](https://ollama.com/download) and download the two models:

&#x20;  ```

&#x20;  ollama pull llama3.2:1b

&#x20;  ollama pull deepseek-r1:1.5b

&#x20;  ```

2\. Create the Python environment and install the packages:

&#x20;  ```

&#x20;  conda create -n HRI python=3.12 -y

&#x20;  conda activate HRI

&#x20;  pip install -r requirements.txt

&#x20;  ```

3\. Install the server's package:

&#x20;  ```

&#x20;  npm install

&#x20;  ```

4\. If your robot's IP address is not `192.168.50.186`, set it:

&#x20;  ```

&#x20;  set REACHY\_HOST=<your robot's IP>

&#x20;  ```



\## Running the robot demo



Start each program in its own terminal, in this order, from the project folder:



```

ollama serve

node wizardOfOz/index.js

python experiment/experiment.py

python wizard\_brain.py

```



Then speak to the robot. Optional settings, set before starting `experiment.py`:



| Setting | Meaning |

|---|---|

| `set REACHY\_TTS\_PROVIDER=local` | use the offline Windows voice instead of Hugging Face |

| `set HF\_TOKEN=<token>` | Hugging Face token, for more voice quota |



Testing tools:



\- `python watch\_messages.py` prints every message passing through the server.

\- `python experiment/record.py` records 5 seconds and plays it back (microphone test).

\- `python experiment/text\_to\_speech.py --text "Hello" --provider local` tests the voice.



\## The detector



The detector uses two complementary signals, both computed from the model itself.



\### 1. Entropy trajectory (`entropy/`)



Token entropy measures how uncertain the model is about each word it writes.

The main idea is that the \*\*shape\*\* of the entropy curve over time matters more

than its level:



\- \*\*Healthy:\*\* entropy rises while the model explores, then falls as it settles.

\- \*\*Overthinking:\*\* entropy stays high or keeps rising, and never settles.



The curve is split into thirds, and the first third (\*\*early\*\*) is compared with

the last third (\*\*late\*\*). The thresholds live in `entropy/common.py`:



| Condition | Verdict |

|---|---|

| Late entropy is low (below 0.6) | \*\*Healthy:\*\* converged or calm |

| Overall entropy is high (above 0.9) and late is still high | \*\*Overthinking:\*\* stayed high |

| Late is clearly higher than early (by more than 0.25) | \*\*Overthinking:\*\* rising |

| None of the above | \*\*Unclear:\*\* the honest middle zone |



| Script | What it does |

|---|---|

| `entropy\_test.py` | Checks that Ollama returns token probabilities, so entropy can be computed |

| `entropy\_trajectory.py` | Runs one question and saves its entropy curve (`entropy\_trajectory.png`) |

| `compare\_cases.py` | \*\*Key result:\*\* an easy question next to an unsolved problem (`compare\_cases.png`) |

| `build\_trial\_bank.py` | Runs six labelled questions three times each and checks the verdicts (`trials/`) |

| `answer\_probe.py` | The answer-peek method below, on its own |

| `common.py` | Shared code: entropy formula, smoothing, and the verdict rules |



Run the scripts from inside the `entropy` folder, since they save their results there.



\*\*Finding:\*\* healthy questions score about 0.0–0.5 and the forced-overthinking

question about 1.4, with an ambiguous band around 0.7–0.8. The Collatz and

even-and-odd questions fall in that band on this small model, which is why the

detector has a third, \*\*unclear\*\* state instead of forcing a yes-or-no answer.



\### 2. Answer probe (`wizard\_brain.py`, `entropy/answer\_probe.py`)



While a reasoning model thinks, the program regularly closes the model's

`<think>` block itself and asks it to finish "The final answer is:". This shows

which answer the model would give if it stopped now.



| State | Meaning |

|---|---|

| \*\*CONVERGED\*\* | the same answer came back twice in a row |

| \*\*UNSETTLED\*\* | the answer kept changing; the most common one is used |

| \*\*STUCK\*\* | no answer could be read at all |



\*\*Important:\*\* converged means \*committed\*, not \*correct\*. On the bat-and-ball

question, the model often locks onto $0.10, the classic wrong answer (the right

answer is $0.05). The detector shows whether the model has stopped changing its

mind; judging whether the answer is right is left to the human supervisor.



\## Limits



\- The detector observes behaviour, not thoughts: LLMs are not interpretable.

\- `llama3.2:1b` is small and noisy, so results vary from run to run; this is why

&#x20; `build\_trial\_bank.py` averages three runs.

\- Entropy is computed over the 10 most likely tokens, so it is an approximation.



\## References



\- Xu et al., \*Unveiling the Entropy Dynamics of Chain-of-Thought Reasoning\* (ICML 2026)

\- Bin/Jiang et al., \*Explore Briefly, Then Decide\* (TECA, 2026)

