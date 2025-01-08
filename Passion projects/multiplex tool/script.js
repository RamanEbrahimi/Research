document.getElementById("upload-btn").addEventListener("click", function () {
    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);

    // Update progress bar during upload
    xhr.upload.onprogress = function (event) {
        if (event.lengthComputable) {
            const percentComplete = Math.round((event.loaded / event.total) * 100);
            const progressBar = document.getElementById("progress-bar");
            progressBar.style.width = percentComplete + "%";
            progressBar.textContent = percentComplete + "%";
        }
    };

    // Handle upload completion
    xhr.onload = function () {
        const status = document.getElementById("status");
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            status.textContent = response.message;
        } else {
            status.textContent = "File upload failed.";
        }
    };

    // Handle errors
    xhr.onerror = function () {
        const status = document.getElementById("status");
        status.textContent = "An error occurred during the upload.";
    };

    // Send the request
    xhr.send(formData);
});
