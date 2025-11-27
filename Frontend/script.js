const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("predictionText");
const progressFill = document.getElementById("progressFill");

let selectedImage = null;

// Preview image
imageInput.addEventListener("change", (e) => {
    selectedImage = e.target.files[0];
    preview.src = URL.createObjectURL(selectedImage);
    preview.style.display = "block";
});

// Predict
predictBtn.addEventListener("click", async () => {
    if (!selectedImage) {
        alert("Please upload an image!");
        return;
    }

    let formData = new FormData();
    formData.append("image", selectedImage);

    loader.style.display = "block";
    resultBox.style.display = "none";

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    loader.style.display = "none";
    resultBox.style.display = "block";

    predictionText.innerText = `Animal: ${data.animal} (${data.confidence}%)`;
    progressFill.style.width = data.confidence + "%";
});
