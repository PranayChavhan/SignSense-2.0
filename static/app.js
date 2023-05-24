$(document).ready(function() {
  var video = document.getElementById('video');
  var startButton = document.getElementById('start-btn');
  var stopButton = document.getElementById('stop-btn');

  startButton.addEventListener('click', function() {
    video.src = "{{ url_for('video_feed') }}";
  });

  stopButton.addEventListener('click', function() {
    video.src = "";
  });
});