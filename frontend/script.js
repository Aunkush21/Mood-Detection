function predictMood() {
    const text = document.getElementById("textInput").value;

    if (text.trim() === "") {
        alert("Please type something!");
        return;
    }

    // UI interactions
    document.getElementById("loader").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
    document.getElementById("result").innerHTML = "";

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        const emoji = {
            happy: "😄",
            sad: "😢",
            angry: "😡",
            neutral: "😐"
        };

        document.getElementById("loader").classList.add("hidden");

        document.getElementById("result").classList.remove("hidden");
        document.getElementById("result").innerHTML =
            `${emoji[data.mood]} ${data.mood.toUpperCase()}`;
    })
    .catch(err => {
        document.getElementById("loader").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");
        document.getElementById("result").innerText = "Error connecting to backend!";
    });
}
