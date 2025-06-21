  let supernovaYear = 1200;
  let observerDistance = 3000;
  let lightArrivalYear = supernovaYear + observerDistance;
  let currentAnimationYear = -3000;
  let animationFrame;
  let animating = false;
  let speedFactor = 5;

  function simulate() {
  supernovaYear = parseInt(document.getElementById("supernovaYear").value);
  lightArrivalYear = supernovaYear + observerDistance;
  currentAnimationYear = supernovaYear;
  animating = false;
  manBurstTriggered = false;

  const star = document.getElementById("starImg");
  const burst = document.getElementById("burstImg");
  const wave = document.getElementById("blastWave");

  // Reset star and burst above the man
  document.getElementById("manStar").style.display = "block";
  document.getElementById("manBurst").style.display = "none";

  // Initial burst + wave reset
  burst.style.display = "block";
  star.style.display = "none";
  wave.style.opacity = 0;
  wave.style.width = `1px`;
  wave.style.height = `1px`;

  // Set timeline to match starting point
  document.getElementById("timeRange").value = currentAnimationYear;
  updateTime(false);  // Sync pointer and year display

  document.getElementById("result").textContent = `Supernova in ${supernovaYear} AD → Observed in ${lightArrivalYear} AD`;

  setTimeout(() => {
    burst.style.display = "none";
    wave.style.opacity = 1;
    animating = true;
    animate();
  }, 2000);
}


  function animate() {
    cancelAnimationFrame(animationFrame);
    const range = document.getElementById("timeRange");
    const pointer = document.getElementById("pointer");
    const wave = document.getElementById("blastWave");

    function step() {
      if (!animating || currentAnimationYear > lightArrivalYear) return;

      range.value = currentAnimationYear;
      document.getElementById("supernovaYear").value = currentAnimationYear;
      document.getElementById("timeDisplay").textContent =
        currentAnimationYear < 0 ? `${-currentAnimationYear} BC` : `${currentAnimationYear} AD`;

      const percent = (currentAnimationYear - parseInt(range.min)) / (parseInt(range.max) - parseInt(range.min));
      pointer.style.left = `${percent * 100}%`;

      const scale = (currentAnimationYear - supernovaYear) / observerDistance;
      const radiusPx = scale * 400;
      wave.style.width = `${radiusPx * 2}px`;
      wave.style.height = `${radiusPx * 2}px`;
      wave.style.left = `${105 - radiusPx + 5}px`;
      wave.style.top = `${105 - radiusPx + 5}px`;

      if (radiusPx >= 400) {
        const manStar = document.getElementById("manStar");
        const manBurst = document.getElementById("manBurst");
        if (manStar.style.display !== "none") {
          manStar.style.display = "none";
          manBurst.style.display = "block";
          setTimeout(() => {
            manBurst.style.display = "none";
          }, 1000);
        }
      }

      if (currentAnimationYear === lightArrivalYear) {
        const resultElement = document.getElementById("result");
        resultElement.textContent = `SN observed at ${lightArrivalYear} AD`;
        resultElement.style.backgroundColor = "green";
        resultElement.style.color = "white";
        resultElement.style.padding = "4px 8px";
        resultElement.style.borderRadius = "4px";
      }

      currentAnimationYear += speedFactor;
      animationFrame = requestAnimationFrame(step);
    }

    step();
  }

  function updateTime(syncInput) {
    const range = document.getElementById("timeRange");
    const currentYear = parseInt(range.value);
    document.getElementById("timeDisplay").textContent =
      currentYear < 0 ? `${-currentYear} BC` : `${currentYear} AD`;

    const pointer = document.getElementById("pointer");
    const percent = (currentYear - parseInt(range.min)) / (parseInt(range.max) - parseInt(range.min));
    pointer.style.left = `${percent * 100}%`;

    if (syncInput) {
      document.getElementById("supernovaYear").value = currentYear;
    }
  }

  function syncTimeWithInput() {
    const inputYear = parseInt(document.getElementById("supernovaYear").value);
    document.getElementById("timeRange").value = inputYear;
    updateTime(false);
  }

  // Initialize pointer position
  window.onload = () => {
    updateTime(false);
  };