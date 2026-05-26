using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Emgu.CV;

namespace chuni_hands {
    public partial class MainWindow : Window {
        private const string ConfigFile = "chuni-hands.json";

        private VideoCapture _capture;
        private readonly List<Sensor> _sensors = new List<Sensor>(6);
        private Mat _mat = new Mat();
        private byte[] _matData = new byte[0];
        private bool _hasPendingReset = false;

        private Task _captureTask;
        private volatile bool _closing = false;
        private volatile bool _isWindowActive = false;

        private readonly Config _config = new Config();
        private readonly HttpClient _http = new HttpClient();

        public Config Config => _config;

        public MainWindow() {
            if (File.Exists(ConfigFile)) {
                _config = Helpers.Deserialize<Config>(ConfigFile);
            }

            for (var i = 0; i < 6; ++i) {
                _sensors.Add(new Sensor(i, _config));
            }

            InitializeComponent();
            Title += " version " + Helpers.GetVersion();

            TheCanvas.Sensors = _sensors;
            RefreshCameras();

            Logger.LogAdded += log => {
                Dispatcher.InvokeAsync(() => { // Ensure logging is thread-safe for UI
                    LogBox.AppendText(log);
                    LogBox.AppendText(Environment.NewLine);
                    LogBox.ScrollToEnd();
                });
            };

            Activated += (sender, e) => _isWindowActive = true;
            Deactivated += (sender, e) => _isWindowActive = false;
        }

        private void ProcessFrame() {
            // compute
            foreach (var sensor in _sensors) {
                sensor.Update(_mat, _hasPendingReset);
            }
            _hasPendingReset = false;

            // send key
            SendKey();
        }

        private readonly object _matLock = new object();

        private void UpdateDisplay() {
            if (!_config.ShowVideo) return;

            int cols, rows, step;

            lock (_matLock) {
                if (_mat.IsEmpty) return;

                cols = _mat.Cols;
                rows = _mat.Rows;
                step = _mat.Step; // FIX: Use OpenCV's actual memory step (stride) instead of math

                var length = rows * step;
                if (_matData.Length < length) {
                    _matData = new byte[length];
                }

                _mat.CopyTo(_matData);
            }

            try {
                // FIX: Pass 'step' (stride) here to prevent WPF crashes on padded images
                var bm = BitmapSource.Create(cols, rows, 96, 96, PixelFormats.Bgr24, null, _matData, step);
                TheCanvas.Image = bm;
                TheCanvas.InvalidateVisual();
            }
            catch (Exception ex) {
                Logger.Error($"Display error: {ex.Message}");
            }
        }


        private void SendKey() {
            if (_isWindowActive) {
                return;
            }

            if (_sensors.All(s => !s.StateChanged)) {
                return;
            }

            switch (_config.SendKeyMode) {
                case "be": {
                    var airKeys = String.Concat(from sensor in _sensors select sensor.Active ? "1" : "0");
                    _http.GetAsync(_config.EndPoint + "?k=" + airKeys);
                    break;
                }
                case "chuni_io": {
                    ChuniIO.Send(_sensors);
                    break;
                }
                default:
                    throw new Exception("unknown SendKeyMode");
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e) {
            StartCapture();
        }

        private void RefreshCameras() {
            var cameras = CameraHelper.CameraHelper.GetCameras();
            CameraCombo.Items.Clear();
            foreach (var cam in cameras) {
                CameraCombo.Items.Add($"[{cam.Id}] {cam.Name}");
            }

            if (_config.CameraId >= 0 && _config.CameraId < CameraCombo.Items.Count) {
                CameraCombo.SelectedIndex = _config.CameraId;
            }
            else {
                _config.CameraId = 0;
            }
        }

        private void StartCapture() {
            // FIX: Try DirectShow (DShow) first. It is vastly more stable on Windows.
            var cap = new VideoCapture(_config.CameraId, VideoCapture.API.DShow);
            if (!cap.IsOpened) {
                // Fallback to Any if DShow fails
                cap = new VideoCapture(_config.CameraId, VideoCapture.API.Any);
            }

            if (!cap.IsOpened) {
                Logger.Error("Failed to start video capture (Camera might be in use by another app)");
                return;
            }

            Logger.Info("Video capture started");
            cap.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, _config.CaptureWidth);
            cap.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, _config.CaptureHeight);
            cap.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.Autofocus, 0);
            cap.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.Exposure, _config.Exposure);

            _capture = cap;

            // FIX: Give the camera a few attempts to warm up.
            // Many cameras return empty frames on the first 3-5 reads.
            bool frameGrabbed = false;
            for (int i = 0; i < 15; i++) {
                _capture.Read(_mat);
                if (!_mat.IsEmpty) {
                    frameGrabbed = true;
                    break;
                }
                Thread.Sleep(100); // Wait 100ms between attempts
            }

            if (!frameGrabbed) {
                Logger.Error("Camera connected, but returned an empty frame. It may not support the requested resolution or format.");
                StopCapture(); // Clean up
                return;
            }

            _config.CaptureWidth = _mat.Cols;
            _config.CaptureHeight = _mat.Rows;

            _captureTask = Task.Run(CaptureLoop);
        }

        private void StopCapture() {
            Logger.Info("Stopping capture");

            _closing = true;
            _captureTask?.Wait(2000); // Prevent infinite deadlock if thread is stuck
            _captureTask = null;

            _capture?.Stop();
            _capture?.Dispose();
            _capture = null;

            _closing = false;
        }

        private void CaptureLoop() {
            var bootstrapFrames = _config.BootstrapSeconds * _config.Fps;
            int emptyFrameCount = 0;

            while (!_closing) {
                bool readAttempted = false;

                if (bootstrapFrames > 0) {
                    lock (_matLock) {
                        _capture.Read(_mat);
                    }
                    --bootstrapFrames;
                }
                else {
                    lock (_matLock) {
                        if (!_config.FreezeVideo) {
                            _capture.Read(_mat);
                            readAttempted = true;
                        }

                        if (!_mat.IsEmpty) {
                            emptyFrameCount = 0;
                            ProcessFrame();
                        }
                    }

                    if (!_mat.IsEmpty) {
                        Dispatcher?.BeginInvoke(new Action(UpdateDisplay));
                    }
                }

                // Detect if camera disconnected or stream died mid-use
                if (readAttempted && _mat.IsEmpty) {
                    emptyFrameCount++;
                    if (emptyFrameCount > 30) {
                        Logger.Error("Camera stream lost (Too many empty frames). Please refresh or reconnect.");
                        break; // Exit loop
                    }
                }

                Thread.Sleep(1000 / _config.Fps);
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
            StopCapture();
            Helpers.Serialize(_config, ConfigFile);
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e) {
            _hasPendingReset = true;
        }

        private void SetThresholdButton_Click(object sender, RoutedEventArgs e) {
            if (Double.TryParse(ThresholdBox.Text, out var v)) {
                _config.Threshold = v;
                Logger.Info($"Threshold = {v}");
            }
            else {
                Logger.Error("Invalid input");
            }
        }

        private void CenterButton_Click(object sender, RoutedEventArgs e) {
            _config.OffsetX = 0;
            _config.OffsetY = 0;
        }

        private void SetCameraBtn_Click(object sender, RoutedEventArgs e) {
            StopCapture();

            _config.CameraId = CameraCombo.SelectedIndex;
            StartCapture();
        }

        private void RefreshCameraBtn_Click(object sender, RoutedEventArgs e) {
            RefreshCameras();
        }
    }
}