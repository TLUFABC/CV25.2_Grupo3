function loadContent(file) {
  const content = document.getElementById('dynamic-content');
  fetch(file)
    .then(response => {
      if (!response.ok) throw new Error('Erro ao carregar arquivo: ' + response.status);
      return response.text();
    })
    .then(data => {
      content.innerHTML = data;
      content.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(error => {
      content.innerHTML = '<p>Erro ao carregar o conteúdo.</p>';
      console.error(error);
    });
}

function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (!section) {
    console.warn('Section not found:', sectionId);
    return;
  }
  const isVisible = section.style.display === 'block';
  section.style.display = isVisible ? 'none' : 'block';
}

