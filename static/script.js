const container = document.getElementById('container-graph')

function cleanContainer() {
  Array.from(container.children).forEach(n => n.remove())
}

function genGraph(data) {
  switch(data.type) {
    case 'column':
      return genColumnGraph(data)
    case 'line':
      return genLineGraph(data)
    default:
      throw new Error(`Unknown graph type: ${data.type}`)
  }
}

function getFormValues() {
  return {
    wt_init_seed: document.getElementById('wt_init_seed').value,
    beta: document.getElementById('beta').value,
    learning_rate: document.getElementById('learning_rate').value,
    nudge1: document.getElementById('nudge1').value,
    nudge2: document.getElementById('nudge2').value,
    n_pyr_layer1: document.getElementById('n_pyr_layer1').value,
    n_pyr_layer2: document.getElementById('n_pyr_layer2').value,
    n_pyr_layer3: document.getElementById('n_pyr_layer3').value,
    self_prediction_steps: document.getElementById('self_prediction_steps').value,
    training_steps: document.getElementById('training_steps').value
  }
}

function handleApiData(json) {
  cleanContainer()

  for (const data of json) {
    const canvas = document.createElement('div')
    container.appendChild(canvas)

    const chart = new ApexCharts(canvas, genGraph(data))
    chart.render()
  }
}

function catchApiErrors(body) {
  cleanContainer()

  const iframe = Object.assign(document.createElement('iframe'), { 'id': 'error' })
  container.appendChild(iframe)

  const iframeDoc = iframe.contentDocument || iframe.contentWindow.document
  iframeDoc.body.innerHTML = body
}

function getData() {
  /**
   *  Fetch tells browser to send a request to the flask backend on the '/data' route.
   */
  const params = new URLSearchParams(getFormValues())
  fetch(`/data?${params}`).then(r => r.ok ? r.json().then(handleApiData) : r.text().then(catchApiErrors))
}
