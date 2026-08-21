(() => {
  const seedInput = document.getElementById("seed-input");
  const predictBtn = document.getElementById("predict-btn");
  const stepsSlider = document.getElementById("steps-slider");
  const stepsValue = document.getElementById("steps-value");
  const quickSeeds = document.getElementById("quick-seeds");
  const status = document.getElementById("console-status");
  const divider = document.getElementById("lane-divider");
  const agreementLabel = document.getElementById("agreement-label");

  const lanes = {
    lstm: {
      track: document.getElementById("lstm-track"),
      output: document.getElementById("lstm-output"),
    },
    rnn: {
      track: document.getElementById("rnn-track"),
      output: document.getElementById("rnn-output"),
    },
  };

  stepsSlider.addEventListener("input", () => {
    stepsValue.textContent = stepsSlider.value;
  });

  quickSeeds.addEventListener("click", (e) => {
    const btn = e.target.closest(".quick-seed");
    if (!btn) return;
    seedInput.value = btn.dataset.seed;
    seedInput.focus();
  });

  seedInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runPrediction();
  });
  predictBtn.addEventListener("click", runPrediction);

  function clearLane(key) {
    lanes[key].track.innerHTML = "";
    lanes[key].output.textContent = "";
  }

  function renderToken(key, step, delayIndex, diverges) {
    const chip = document.createElement("span");
    chip.className = "token-chip" + (diverges ? " diverges" : "");
    chip.style.animationDelay = `${delayIndex * 90}ms`;

    const word = document.createElement("span");
    word.textContent = step.word;
    chip.appendChild(word);

    const bar = document.createElement("span");
    bar.className = "conf-bar";
    const topProb = step.candidates[0]?.probability ?? 0;
    bar.style.width = `${Math.max(6, Math.round(topProb * 100))}%`;
    chip.appendChild(bar);

    const pct = Math.round(topProb * 100);
    const runnerUps = step.candidates
      .slice(1)
      .map((c) => `${c.word} (${Math.round(c.probability * 100)}%)`)
      .join(", ");
    chip.title = `${pct}% confident` + (runnerUps ? ` — next best: ${runnerUps}` : "");

    lanes[key].track.appendChild(chip);
  }

  async function runPrediction() {
    const text = seedInput.value.trim();
    if (!text) {
      status.textContent = "Type some seed text first.";
      status.classList.add("is-error");
      seedInput.focus();
      return;
    }
    status.classList.remove("is-error");

    const steps = parseInt(stepsSlider.value, 10);

    predictBtn.disabled = true;
    predictBtn.querySelector(".run-btn-label").textContent = "Thinking…";
    status.textContent = "Running both models…";
    divider.className = "lane-divider";
    agreementLabel.textContent = "— comparing —";

    clearLane("lstm");
    clearLane("rnn");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, steps }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Prediction failed.");
      }

      const data = await res.json();

      const lstmTrace = data.lstm.trace;
      const rnnTrace = data.rnn.trace;
      const len = Math.max(lstmTrace.length, rnnTrace.length);

      let agreements = 0;
      let compared = 0;

      for (let i = 0; i < len; i++) {
        const lstmStep = lstmTrace[i];
        const rnnStep = rnnTrace[i];
        const diverges = lstmStep && rnnStep && lstmStep.word !== rnnStep.word;
        if (lstmStep && rnnStep) {
          compared++;
          if (!diverges) agreements++;
        }
        if (lstmStep) renderToken("lstm", lstmStep, i, diverges);
        if (rnnStep) renderToken("rnn", rnnStep, i, diverges);
      }

      lanes.lstm.output.textContent = data.lstm.full_text;
      lanes.rnn.output.textContent = data.rnn.full_text;

      if (compared === 0) {
        agreementLabel.textContent = "— no comparable tokens —";
      } else if (agreements === compared) {
        divider.classList.add("agree");
        agreementLabel.textContent = `both models agreed on all ${compared} word${compared > 1 ? "s" : ""}`;
      } else if (agreements === 0) {
        divider.classList.add("disagree");
        agreementLabel.textContent = `models disagreed on every word`;
      } else {
        divider.classList.add("disagree");
        agreementLabel.textContent = `agreed on ${agreements} of ${compared} words`;
      }

      status.textContent = "Done.";
    } catch (err) {
      status.textContent = err.message || "Something went wrong.";
      status.classList.add("is-error");
      agreementLabel.textContent = "— error —";
    } finally {
      predictBtn.disabled = false;
      predictBtn.querySelector(".run-btn-label").textContent = "Run inference";
    }
  }
})();
