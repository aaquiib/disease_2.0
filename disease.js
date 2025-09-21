function previewImage(input) {
    const preview = document.getElementById('previewImage');
    const file = input.files[0];
    const reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
        preview.style.display = 'none';
    }
}

async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('className').innerText = `Class: ${data.class}`;

        // Adjust confidence (if backend sends 0–1, multiply by 100)
        let confidence = data.confidence > 1 ? data.confidence : data.confidence * 100;
        document.getElementById('confidence').innerText = `Confidence: ${confidence.toFixed(2)}%`;
        
        // Scroll to the results section
        const resultsSection = document.querySelector('.results');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error:', error);
        alert(`Error uploading image: ${error.message}`);
    }
}

// Add event listener to file input for image preview
document.getElementById('imageUpload').addEventListener('change', function() {
    previewImage(this);
});

// Prevent default form submission
document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();
    uploadImage();
});