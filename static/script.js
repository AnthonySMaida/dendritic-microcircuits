function generateImgNodeFromBase64(b64) {
  return Object.assign(document.createElement('img'), { src: `data:image/png;base64,${b64}`})
}

function getGraphs() {
  fetch('/', { method: 'POST' }).then(response => response.json())
    .then(data => {
      const container = document.getElementById('container')

      container.appendChild(generateImgNodeFromBase64(data.data1))
      container.appendChild(generateImgNodeFromBase64(data.data2))
    })
    .catch(error => console.error(error))
}

function cleanGraphs() {
  Array.from(document.getElementById('container').children).forEach(n => n.remove())
}
