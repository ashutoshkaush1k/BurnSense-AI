// ---------------------------------------------------------------------------
// Backend API
// ---------------------------------------------------------------------------
const API_URL = "http://localhost:8000/analyze";

let activeResult = null;

// ---------------------------------------------------------------------------
// Theme (default dark, explicit toggle overrides)
// ---------------------------------------------------------------------------
const themeToggleBtn = document.getElementById("theme-toggle");

function applyStoredTheme() {
  const stored = localStorage.getItem("burnsense-theme");
  const isDark = stored ? stored === "dark" : true;
  document.documentElement.classList.toggle("dark", isDark);
}

themeToggleBtn.addEventListener("click", () => {
  const isDark = document.documentElement.classList.toggle("dark");
  localStorage.setItem("burnsense-theme", isDark ? "dark" : "light");
});

applyStoredTheme();

// ---------------------------------------------------------------------------
// Cursor-tracking spotlight: one global listener updates root-level CSS custom
// properties consumed by #cursor-spotlight's radial-gradient in styles.css.
// ---------------------------------------------------------------------------
document.addEventListener("mousemove", (e) => {
  document.documentElement.style.setProperty("--mouse-x", e.clientX);
  document.documentElement.style.setProperty("--mouse-y", e.clientY);
});

// ---------------------------------------------------------------------------
// Screen transitions (Landing <-> Diagnostic)
// ---------------------------------------------------------------------------
const landingScreen = document.getElementById("landing-screen");
const chatScreen = document.getElementById("chat-screen");
const startDiagnosisBtn = document.getElementById("start-diagnosis-btn");
const backBtn = document.getElementById("back-btn");

function showChatScreen() {
  landingScreen.classList.add("opacity-0", "-translate-y-4", "pointer-events-none");
  setTimeout(() => {
    landingScreen.classList.add("hidden");
    chatScreen.classList.remove("hidden");
    chatScreen.classList.add("flex");
    setTimeout(() => chatScreen.classList.remove("opacity-0"), 20);
    if (!chatMessages.children.length) sendGreeting();
  }, 300);
}

function showLandingScreen() {
  chatScreen.classList.add("opacity-0");
  setTimeout(() => {
    chatScreen.classList.add("hidden");
    chatScreen.classList.remove("flex");
    landingScreen.classList.remove("hidden");
    setTimeout(() => landingScreen.classList.remove("opacity-0", "-translate-y-4", "pointer-events-none"), 20);
  }, 300);
}

startDiagnosisBtn.addEventListener("click", showChatScreen);
backBtn.addEventListener("click", showLandingScreen);

// ---------------------------------------------------------------------------
// Result copy
// ---------------------------------------------------------------------------
const DEGREE_SUMMARIES = {
  "1st_degree": "This appears to be a superficial 1st-degree burn. It should heal on its own in a few days. Keep it cool and clean.",
  "2nd_degree": "This looks like a 2nd-degree burn, reaching deeper skin layers. It may blister and takes longer to heal — keep it clean and covered, and consider seeing a doctor.",
  "3rd_degree": "This appears to be a severe 3rd-degree burn affecting deep tissue. This requires immediate emergency medical attention — do not treat this at home.",
};

// ---------------------------------------------------------------------------
// Diagnostic feed
// ---------------------------------------------------------------------------
const chatMessages = document.getElementById("chat-messages");
const uploadPill = document.getElementById("upload-pill");
const imageUploadInput = document.getElementById("image-upload-input");

const FLAT_BUBBLE_CLASS = "bg-white dark:bg-[#1E1E1E] border border-neutral-200 dark:border-neutral-800";

function scrollChatToBottom() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

function sendGreeting() {
  const el = document.createElement("div");
  el.className = `animate-slideUpFade max-w-xl rounded-3xl ${FLAT_BUBBLE_CLASS} px-5 py-4 text-sm leading-relaxed`;
  el.textContent = "Hi — tap the button below to upload a clear, well-lit photo of the burn and I'll analyze the boundary, degree, and severity for you.";
  chatMessages.appendChild(el);
}

function appendUserImageMessage(imageDataUrl) {
  const wrapper = document.createElement("div");
  wrapper.className = "animate-slideUpFade flex justify-end";

  const img = document.createElement("img");
  img.src = imageDataUrl;
  img.alt = "Uploaded burn photo";
  img.className = "max-w-md max-h-72 w-auto object-cover rounded-2xl";

  wrapper.appendChild(img);
  chatMessages.appendChild(wrapper);
  scrollChatToBottom();
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "animate-slideUpFade";
  wrapper.id = "typing-indicator";
  wrapper.innerHTML = `
    <div class="inline-flex items-center gap-1.5 rounded-3xl ${FLAT_BUBBLE_CLASS} px-4 py-3">
      <span class="w-2 h-2 rounded-full bg-neutral-500 animate-bounce" style="animation-delay:0ms"></span>
      <span class="w-2 h-2 rounded-full bg-neutral-500 animate-bounce" style="animation-delay:150ms"></span>
      <span class="w-2 h-2 rounded-full bg-neutral-500 animate-bounce" style="animation-delay:300ms"></span>
    </div>`;
  chatMessages.appendChild(wrapper);
  scrollChatToBottom();
}

function removeTypingIndicator() {
  document.getElementById("typing-indicator")?.remove();
}

function degreeLabel(key) {
  return key.replace("_", " ");
}

function topDegree(classProbs) {
  return Object.entries(classProbs).sort((a, b) => b[1] - a[1])[0];
}

function appendSummaryBubble(topKey) {
  const summary = DEGREE_SUMMARIES[topKey] ?? "Analysis complete — see the clinical graphs below for the full breakdown.";

  const wrapper = document.createElement("div");
  wrapper.className = `animate-slideUpFade max-w-xl rounded-3xl ${FLAT_BUBBLE_CLASS} p-4`;
  wrapper.innerHTML = `
    <p class="font-display font-semibold text-base capitalize mb-1">${degreeLabel(topKey)} burn detected</p>
    <p class="text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed">Summary: ${summary}</p>
  `;
  chatMessages.appendChild(wrapper);
  scrollChatToBottom();
}

function appendAnalyticsButtonBubble(result) {
  const wrapper = document.createElement("div");
  wrapper.className = "animate-slideUpFade max-w-xl";
  wrapper.innerHTML = `
    <button class="view-analytics-btn w-full flex items-center justify-center gap-2 rounded-3xl ${FLAT_BUBBLE_CLASS} px-4 py-3 text-sm font-medium hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-all duration-[400ms] ease-in-out">
      <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19V9m6 10V5m6 14v-7" stroke-linecap="round" stroke-linejoin="round"/></svg>
      View Clinical Graphs and Analytics
    </button>
  `;
  wrapper.querySelector(".view-analytics-btn").addEventListener("click", () => openAnalyticsModal(result));
  chatMessages.appendChild(wrapper);
  scrollChatToBottom();
}

function appendErrorCard(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "animate-slideUpFade max-w-xl rounded-3xl bg-red-50 dark:bg-red-950/40 border border-red-300 dark:border-red-900 px-5 py-4";
  wrapper.innerHTML = `
    <p class="font-display font-semibold text-red-700 dark:text-red-300 mb-1">Invalid Image Detected</p>
    <p class="error-message-text text-sm text-red-700/90 dark:text-red-300/80 leading-relaxed"></p>
  `;
  wrapper.querySelector(".error-message-text").textContent = message;
  chatMessages.appendChild(wrapper);
  scrollChatToBottom();
}

async function analyzeImage(file, displayDataUrl) {
  appendTypingIndicator();

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(API_URL, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`Server responded with ${response.status}`);

    const data = await response.json();
    removeTypingIndicator();

    if (data.status === "error") {
      appendErrorCard(data.message);
      return;
    }

    const result = {
      class_probs: data.class_probs,
      area_fraction: data.area,
      estimated_days: data.estimated_days,
      bsi_score: data.bsi_score,
      infection_risk: data.infection_risk,
      action_plan: data.action_plan,
    };
    const [topKey] = topDegree(result.class_probs);

    appendSummaryBubble(topKey);
    appendAnalyticsButtonBubble(result);
  } catch (err) {
    removeTypingIndicator();
    appendErrorCard("Couldn't reach the analysis server. Make sure run_app.bat is running, then try again.");
    console.error("Analysis request failed:", err);
  }
}

function handleImageSelected(file) {
  if (!file || !file.type.startsWith("image/")) return;

  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;
    appendUserImageMessage(dataUrl);
    analyzeImage(file, dataUrl);
  };
  reader.readAsDataURL(file);
}

uploadPill.addEventListener("click", () => imageUploadInput.click());

imageUploadInput.addEventListener("change", () => {
  const file = imageUploadInput.files[0];
  handleImageSelected(file);
  imageUploadInput.value = "";
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadPill.addEventListener(eventName, (e) => {
    e.preventDefault();
    uploadPill.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadPill.addEventListener(eventName, (e) => {
    e.preventDefault();
    uploadPill.classList.remove("drag-active");
  });
});

uploadPill.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  handleImageSelected(file);
});

// ---------------------------------------------------------------------------
// Analytics modal
// ---------------------------------------------------------------------------
const analyticsModal = document.getElementById("analytics-modal");
const modalPanel = document.getElementById("modal-panel");
const modalBackdrop = document.getElementById("modal-backdrop");
const closeModalBtn = document.getElementById("close-modal-btn");

const gaugeRing = document.getElementById("gauge-ring");
const gaugeValue = document.getElementById("gauge-value");
const gaugeRiskBadge = document.getElementById("gauge-risk-badge");
const infectionRiskValue = document.getElementById("infection-risk-value");
const healingDaysValue = document.getElementById("healing-days-value");
const actionPlanValue = document.getElementById("action-plan-value");

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 58;

const RISK_STYLES = {
  Low: { ring: "stroke-emerald-500", badge: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300" },
  Medium: { ring: "stroke-amber-500", badge: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" },
  High: { ring: "stroke-red-500", badge: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" },
};

function openAnalyticsModal(result) {
  activeResult = result;
  const riskStyle = RISK_STYLES[result.infection_risk] ?? RISK_STYLES.Medium;

  gaugeRing.setAttribute("stroke-dasharray", GAUGE_CIRCUMFERENCE.toFixed(2));
  gaugeRing.setAttribute("stroke-dashoffset", GAUGE_CIRCUMFERENCE.toFixed(2));
  gaugeRing.classList.remove("stroke-emerald-500", "stroke-amber-500", "stroke-red-500");
  gaugeValue.textContent = "0";

  gaugeRiskBadge.className = `mt-4 px-3 py-1 rounded-full text-xs font-medium ${riskStyle.badge}`;
  gaugeRiskBadge.textContent = `${result.infection_risk} risk`;
  infectionRiskValue.textContent = result.infection_risk;
  healingDaysValue.textContent = `${result.estimated_days} days`;
  actionPlanValue.textContent = result.action_plan;

  analyticsModal.classList.remove("hidden");
  analyticsModal.classList.add("flex");

  setTimeout(() => {
    gaugeRing.classList.add(riskStyle.ring);
    const offset = GAUGE_CIRCUMFERENCE * (1 - result.bsi_score / 100);
    gaugeRing.setAttribute("stroke-dashoffset", offset.toFixed(2));
    animateCountUp(gaugeValue, result.bsi_score);
  }, 20);
}

function animateCountUp(el, target) {
  const duration = 900;
  const stepMs = 16;
  const steps = Math.round(duration / stepMs);
  let currentStep = 0;

  const intervalId = setInterval(() => {
    currentStep += 1;
    const progress = Math.min(currentStep / steps, 1);
    el.textContent = Math.round(progress * target);
    if (progress >= 1) clearInterval(intervalId);
  }, stepMs);
}

function closeAnalyticsModal() {
  modalPanel.classList.add("opacity-0", "scale-95");
  analyticsModal.classList.add("opacity-0");
  setTimeout(() => {
    analyticsModal.classList.add("hidden");
    analyticsModal.classList.remove("flex", "opacity-0");
    modalPanel.classList.remove("opacity-0", "scale-95");
  }, 250);
}

closeModalBtn.addEventListener("click", closeAnalyticsModal);
modalBackdrop.addEventListener("click", closeAnalyticsModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !analyticsModal.classList.contains("hidden")) closeAnalyticsModal();
});
