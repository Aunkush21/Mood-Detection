// MoodSense frontend — talks to the Flask /predict endpoint and themes the
// whole UI based on the detected mood. Only known moods are ever rendered, and
// all server-derived text goes in via textContent (never innerHTML).

const API_URL = "http://127.0.0.1:5000/predict";

const MOODS = {
    happy:   { emoji: "😄", caption: "Bright and upbeat — there's real joy in this." },
    sad:     { emoji: "😢", caption: "A heavier, downcast tone comes through here." },
    angry:   { emoji: "😡", caption: "Sharp and frustrated — there's some heat in these words." },
    neutral: { emoji: "😐", caption: "Calm and matter-of-fact, with no strong emotion." },
};

const el = (id) => document.getElementById(id);

function setLoading(isLoading) {
    el("analyzeBtn").classList.toggle("loading", isLoading);
}

function showError(message) {
    const box = el("errorMsg");
    box.textContent = message;
    box.hidden = false;
}

function clearError() {
    el("errorMsg").hidden = true;
}

function renderMood(mood) {
    const info = MOODS[mood];           // whitelist: unknown labels are rejected
    if (!info) {
        showError("Got an unexpected response from the model.");
        return;
    }

    document.documentElement.setAttribute("data-mood", mood);

    el("moodEmoji").textContent = info.emoji;
    el("moodLabel").textContent = mood.toUpperCase();
    el("moodCaption").textContent = info.caption;

    const result = el("result");
    result.hidden = false;
    result.classList.remove("show");
    void result.offsetWidth;            // restart the entrance animation
    result.classList.add("show");
}

async function predictMood() {
    const text = el("textInput").value.trim();
    clearError();

    if (text === "") {
        showError("Please type something to analyze first.");
        return;
    }

    setLoading(true);
    el("result").hidden = true;

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) throw new Error(`Server responded ${res.status}`);

        const data = await res.json();
        renderMood(data.mood);
    } catch (err) {
        document.documentElement.setAttribute("data-mood", "idle");
        showError("Couldn't reach the backend. Is the API running on port 5000?");
    } finally {
        setLoading(false);
    }
}

// Ctrl/Cmd + Enter to analyze.
document.getElementById("textInput").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        predictMood();
    }
});
