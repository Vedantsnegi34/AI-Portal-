function applyJob(button) {
  button.textContent = 'Applied';
  button.disabled = true;
  alert('Your application has been submitted.');
}

document.getElementById('loginForm')?.addEventListener('submit', function(e) {
  e.preventDefault();
  alert('Login submitted! (You can add real validation or redirection here)');
});