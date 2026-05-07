document.addEventListener('DOMContentLoaded', () => {
    const enrollForm = document.getElementById('enrollForm');
    const enrollBtn = document.getElementById('enrollBtn');
    const dropZone = document.getElementById('dropZone');
    const identifyFile = document.getElementById('identifyFile');
    const resultDiv = document.getElementById('result');

    // Maneuver Drop Zone
    dropZone.addEventListener('click', () => identifyFile.click());
    
    identifyFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleIdentify(e.target.files[0]);
        }
    });

    // Handle Enrollment
    enrollForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        formData.append('name', document.getElementById('name').value);
        formData.append('identity_number', document.getElementById('id_number').value);
        formData.append('eye_side', document.getElementById('eye_side').value);
        formData.append('file', document.getElementById('enrollFile').files[0]);

        enrollBtn.disabled = true;
        enrollBtn.innerHTML = '<span class="loader"></span> Registering...';

        try {
            const response = await fetch('/enroll', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            alert(result.message || result.detail);
            enrollForm.reset();
        } catch (error) {
            alert('Error connecting to server');
        } finally {
            enrollBtn.disabled = false;
            enrollBtn.textContent = 'Register Biometrics';
        }
    });

    // Handle Identification
    async function handleIdentify(file) {
        const formData = new FormData();
        formData.append('file', file);

        resultDiv.style.display = 'block';
        resultDiv.className = 'card'; // Reset
        resultDiv.innerHTML = '<span class="loader"></span> Scanning Iris Pattern...';

        try {
            const response = await fetch('/identify', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.status === 'matched') {
                resultDiv.className = 'success';
                resultDiv.innerHTML = `
                    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Access Granted</div>
                    <div style="color: white; font-size: 1.5rem;">${data.user}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">ID: ${data.identity} | Confidence: ${((1-data.distance)*100).toFixed(1)}%</div>
                `;
            } else {
                resultDiv.className = 'error';
                resultDiv.innerHTML = `
                    <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Access Denied</div>
                    <div style="font-size: 0.8rem;">Pattern not recognized (Dist: ${data.distance})</div>
                `;
            }
        } catch (error) {
            resultDiv.className = 'error';
            resultDiv.textContent = 'Error connecting to server';
        }
    }
});
