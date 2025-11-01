# FFT and Signal Processing - Practical Assignments Guide

**Module 3: Signal Processing Fundamentals - Day 1-2 Practical Work**  
**Course:** Complete Computer Vision Engineer - MIT Signal Processing Track  
**Duration:** 6-8 hours of focused implementation  
**Prerequisites:** MIT 6.003 Lectures on Fourier Analysis, Basic Python/NumPy

---

## Overview

This guide provides 10 progressive practical assignments designed to cement your theoretical understanding of FFT and signal processing concepts from MIT's course. Each assignment builds upon previous concepts while advancing toward computer vision applications.

**Learning Objectives:**
- Master FFT implementation and properties
- Understand frequency domain analysis and filtering
- Bridge 1D signal processing to 2D image processing
- Build practical filtering systems
- Connect concepts to computer vision applications

---

## Foundation Level: FFT Mechanics (Assignments 1-3)

### Assignment 1: FFT Fundamentals and Properties
**Duration:** 45-60 minutes  
**Difficulty:** ⭐⭐☆☆☆

#### Objectives
- Build intuition for what FFT actually computes
- Verify theoretical properties with code
- Compare manual DFT implementation with NumPy FFT

#### Tasks
1. **Implement DFT from scratch** for N=8, 16 using the mathematical definition:
   ```python
   X[k] = Σ(n=0 to N-1) x[n] * e^(-j*2π*k*n/N)
   ```

2. **Generate test signals:**
   - Pure sinusoid: `x[n] = cos(2π*f*n/fs)`
   - Complex exponential: `x[n] = e^(j*2π*f*n/fs)`
   - Sum of sinusoids with different frequencies

3. **Verify FFT properties:**
   - Linearity: FFT(ax + by) = a*FFT(x) + b*FFT(y)
   - Time shift: FFT(x[n-k]) = e^(-j*2π*k*m/N) * FFT(x[n])
   - Frequency shift: FFT(x[n]*e^(j*2π*k*n/N)) = X[m-k]
   - Parseval's theorem: Energy in time = Energy in frequency

4. **Visualization requirements:**
   - Plot magnitude and phase spectra
   - Compare your DFT vs NumPy FFT results
   - Show before/after for each property verification

#### Expected Output
- Functions: `manual_dft()`, `verify_linearity()`, `verify_time_shift()`
- Plots: Magnitude/phase spectra comparisons
- Verification: All properties should match theoretical predictions within numerical precision

#### Key Insights to Gain
- FFT is just an efficient way to compute DFT
- Phase information is as important as magnitude
- Discrete frequency bins correspond to specific frequencies

---

### Assignment 2: Frequency Domain Signal Analysis
**Duration:** 60-75 minutes  
**Difficulty:** ⭐⭐⭐☆☆

#### Objectives
- Master frequency domain interpretation of real signals
- Understand windowing and spectral leakage
- Build intuition for spectrograms and time-frequency analysis

#### Tasks
1. **Signal generation and analysis:**
   ```python
   # Create composite signal
   fs = 1000  # Sample rate
   t = np.linspace(0, 2, 2*fs)
   signal = (2*np.cos(2*np.pi*50*t) + 
            0.5*np.cos(2*np.pi*120*t) + 
            0.1*np.random.randn(len(t)))
   ```

2. **Spectral analysis:**
   - Compute and plot magnitude spectrum
   - Identify frequency peaks and relate to original components
   - Calculate signal power in different frequency bands
   - Analyze effect of signal length on frequency resolution

3. **Windowing investigation:**
   - Apply different windows: rectangular, Hann, Hamming, Blackman
   - Compare spectral leakage for each window
   - Demonstrate window choice trade-offs (main lobe vs side lobes)

4. **Time-varying signals:**
   - Create chirp signal (frequency sweeping over time)
   - Implement short-time FFT (spectrogram)
   - Analyze signals with time-varying frequency content

#### Expected Output
- Function: `analyze_spectrum()`, `apply_windowing()`, `compute_spectrogram()`
- Plots: Spectrum plots, window comparisons, spectrograms
- Analysis: Written comparison of different windowing effects

#### Key Insights to Gain
- Real signals have noise and multiple frequency components
- Windowing is essential for analyzing finite-length signals
- Time-frequency trade-off in spectral analysis

---

### Assignment 3: Convolution via FFT (1D)
**Duration:** 60-75 minutes  
**Difficulty:** ⭐⭐⭐☆☆

#### Objectives
- Verify and implement the convolution theorem
- Understand computational complexity advantages of FFT
- Handle practical issues: zero-padding, circular convolution

#### Tasks
1. **Direct convolution implementation:**
   ```python
   def direct_convolution(x, h):
       # Implement y[n] = Σ x[k]h[n-k] directly
       pass
   ```

2. **FFT-based convolution:**
   ```python
   def fft_convolution(x, h):
       # Implement using: IFFT(FFT(x) * FFT(h))
       # Handle zero-padding for linear convolution
       pass
   ```

3. **Test signals:**
   - Input: rectangular pulse, exponential decay, Gaussian pulse
   - Impulse responses: low-pass filter, differentiator, moving average

4. **Comparison and validation:**
   - Compare outputs of both methods (should be identical)
   - Measure execution time for different signal lengths
   - Plot computational complexity (N log N vs N²)
   - Handle edge cases and circular convolution artifacts

5. **Advanced challenge:**
   - Implement overlap-add method for long signals
   - Demonstrate real-time filtering simulation

#### Expected Output
- Functions: `direct_convolution()`, `fft_convolution()`, `overlap_add()`
- Plots: Convolution results, timing comparisons, complexity analysis
- Validation: Numerical accuracy verification between methods

#### Key Insights to Gain
- Convolution theorem enables efficient filtering
- Zero-padding is crucial for linear convolution
- FFT becomes advantageous for longer signals

---

## Intermediate Level: 1D Filtering (Assignments 4-5)

### Assignment 4: Digital Filter Design and Implementation
**Duration:** 75-90 minutes  
**Difficulty:** ⭐⭐⭐⭐☆

#### Objectives
- Design filters in frequency domain
- Compare frequency domain vs time domain implementations
- Analyze filter characteristics and performance

#### Tasks
1. **Frequency domain filter design:**
   ```python
   def design_frequency_filter(N, fs, filter_type, cutoff):
       # Create ideal filters by manipulating FFT bins
       # Types: 'lowpass', 'highpass', 'bandpass', 'bandstop'
       pass
   ```

2. **Filter implementations:**
   - Ideal filters (brick-wall response)
   - Butterworth filters (smooth rolloff)
   - Gaussian filters (no ringing)
   - Custom filter shapes

3. **Time domain comparison:**
   - Use SciPy's `scipy.signal.butter()`, `scipy.signal.filtfilt()`
   - Compare frequency responses using `scipy.signal.freqz()`
   - Analyze impulse responses and stability

4. **Filter analysis:**
   - Measure cutoff frequency, rolloff rate, ripple
   - Test with known signals and noise
   - Demonstrate filter artifacts (ringing, phase distortion)

5. **Performance evaluation:**
   - SNR improvement measurements
   - Frequency response accuracy
   - Computational efficiency comparison

#### Expected Output
- Functions: `design_frequency_filter()`, `analyze_filter_response()`
- Plots: Filter responses, before/after filtering, impulse responses
- Analysis: Quantitative comparison of different filter types

#### Key Insights to Gain
- Frequency domain design offers direct control over response
- Trade-offs between filter sharpness and artifacts
- Understanding of filter terminology and specifications

#### Resources
- SciPy Signal Processing: `scipy.signal` module documentation
- Focus on: `freqz()`, `butter()`, `filtfilt()`, `hilbert()`

---

### Assignment 5: Audio Signal Processing
**Duration:** 75-90 minutes  
**Difficulty:** ⭐⭐⭐⭐☆

#### Objectives
- Apply FFT filtering to real-world audio signals
- Implement practical audio processing effects
- Bridge signal processing theory to applications

#### Tasks
1. **Audio loading and analysis:**
   ```python
   import librosa  # or scipy.io.wavfile
   
   # Load audio file
   audio, sr = librosa.load('audio_file.wav', sr=None)
   
   # Analyze spectrum
   analyze_audio_spectrum(audio, sr)
   ```

2. **Noise reduction system:**
   - Identify noise frequency characteristics
   - Design notch filters for specific frequency removal
   - Implement spectral subtraction algorithm
   - Compare different noise reduction approaches

3. **Audio effects implementation:**
   - **Echo/Delay:** Implement using convolution with delayed impulse
   - **Reverb:** Create impulse response, apply via convolution
   - **Pitch shifting:** Modify frequency domain representation
   - **Equalization:** Multi-band filtering system

4. **Advanced processing:**
   - Implement simple vocoder (phase vocoder)
   - Time-stretching without pitch change
   - Harmonic-percussive separation

#### Expected Output
- Functions: `remove_noise()`, `add_echo()`, `pitch_shift()`
- Audio files: Before/after processing examples
- Plots: Spectrograms showing processing effects
- Demo: Interactive audio processing tool

#### Key Insights to Gain
- Real signals have complex spectral characteristics
- Phase information crucial for audio quality
- Trade-offs between processing quality and computational cost

#### Resources
- Librosa documentation for audio processing
- PyImageSearch audio processing tutorials
- Audio effect algorithms and implementations

---

## Advanced Level: 2D Image Processing (Assignments 6-8)

### Assignment 6: 2D FFT and Image Frequency Analysis
**Duration:** 90-105 minutes  
**Difficulty:** ⭐⭐⭐⭐☆

#### Objectives
- Extend FFT concepts to 2D images
- Understand image frequency domain characteristics
- Master 2D frequency domain visualization

#### Tasks
1. **2D FFT implementation and visualization:**
   ```python
   def analyze_image_spectrum(image):
       # Compute 2D FFT
       fft_image = np.fft.fft2(image)
       fft_shifted = np.fft.fftshift(fft_image)
       
       # Visualize magnitude and phase
       magnitude = np.log(np.abs(fft_shifted) + 1)
       phase = np.angle(fft_shifted)
       
       return magnitude, phase
   ```

2. **Image types analysis:**
   - Natural images (photos, textures)
   - Synthetic images (geometric patterns)
   - Test patterns (checkerboard, sinusoidal gratings)
   - Compare frequency domain characteristics

3. **Frequency domain interpretation:**
   - Identify DC component (average intensity)
   - Low frequency regions (smooth variations)
   - High frequency regions (edges, details)
   - Directional frequency content

4. **Phase vs magnitude importance:**
   - Reconstruct images using only magnitude
   - Reconstruct images using only phase
   - Swap magnitude/phase between different images
   - Demonstrate phase importance for perception

#### Expected Output
- Functions: `analyze_image_spectrum()`, `reconstruct_from_components()`
- Plots: 2D FFT visualizations, phase/magnitude comparisons
- Analysis: Report on different image types' spectral characteristics

#### Key Insights to Gain
- Images have predictable frequency domain structure
- Phase contains most spatial information
- Different image types have distinct spectral signatures

---

### Assignment 7: Image Filtering in Frequency Domain
**Duration:** 90-105 minutes  
**Difficulty:** ⭐⭐⭐⭐⭐

#### Objectives
- Master 2D convolution via FFT
- Implement practical image filters
- Compare spatial vs frequency domain efficiency

#### Tasks
1. **Frequency domain filter implementation:**
   ```python
   def frequency_domain_filter(image, filter_func):
       # Apply filter in frequency domain
       fft_image = np.fft.fft2(image)
       fft_shifted = np.fft.fftshift(fft_image)
       
       # Create frequency domain filter
       H = filter_func(fft_image.shape)
       
       # Apply filter and inverse transform
       filtered_fft = fft_shifted * H
       filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
       
       return filtered_image
   ```

2. **Filter types to implement:**
   - **Low-pass filters:** Gaussian, Butterworth, Ideal
   - **High-pass filters:** For edge enhancement
   - **Band-pass/Band-stop:** For specific frequency ranges
   - **Custom filters:** Directional, ring filters

3. **Classical image processing operations:**
   - Gaussian blur (compare with cv2.GaussianBlur)
   - Unsharp masking for sharpening
   - Laplacian edge detection
   - Motion blur simulation and removal

4. **Performance comparison:**
   - Measure execution time: spatial vs frequency filtering
   - Analyze crossover point where FFT becomes faster
   - Handle different image sizes and filter sizes

5. **Boundary effects handling:**
   - Zero padding vs reflection padding
   - Circular convolution artifacts
   - Windowing approaches for images

#### Expected Output
- Functions: `frequency_domain_filter()`, `compare_filter_methods()`
- Plots: Filter responses, before/after images, timing comparisons
- Analysis: When to use frequency vs spatial domain filtering

#### Key Insights to Gain
- Frequency domain filtering offers different advantages
- Large kernel operations benefit most from FFT
- Boundary handling is crucial for good results

#### Resources
- OpenCV documentation for comparison operations
- SciPy ndimage for spatial domain operations
- NumPy FFT for frequency domain operations

---

### Assignment 8: Advanced Image Processing Techniques
**Duration:** 105-120 minutes  
**Difficulty:** ⭐⭐⭐⭐⭐

#### Objectives
- Tackle complex filtering scenarios
- Implement restoration and enhancement algorithms
- Bridge to computer vision preprocessing

#### Tasks
1. **Deconvolution for motion blur removal:**
   ```python
   def remove_motion_blur(blurred_image, blur_kernel):
       # Implement Wiener deconvolution
       # Handle division by small numbers in frequency domain
       pass
   ```

2. **Homomorphic filtering:**
   - Separate illumination and reflectance components
   - Apply different filtering to each component
   - Useful for uneven lighting correction

3. **Directional filtering:**
   - Design filters that respond to specific orientations
   - Implement Gabor filters in frequency domain
   - Edge detection in specific directions

4. **Noise reduction systems:**
   - Wiener filtering for known noise characteristics
   - Spectral subtraction methods
   - Compare with spatial domain methods (bilateral filter)

5. **Advanced applications:**
   - Periodic noise removal (power line interference)
   - Texture enhancement/suppression
   - Pattern detection using template matching in frequency domain

#### Expected Output
- Functions: `wiener_deconvolve()`, `homomorphic_filter()`, `directional_filter()`
- Demonstrations: Before/after results on challenging images
- Analysis: Comparison with modern deep learning approaches

#### Key Insights to Gain
- Classical methods still relevant for many applications
- Understanding limitations and when to use each technique
- Foundation for understanding CNN filter learning

#### Resources
- PyImageSearch advanced filtering tutorials
- Digital Image Processing textbook examples
- Research papers on restoration algorithms

---

## Expert Level: Real-world Applications (Assignments 9-10)

### Assignment 9: Multi-scale Image Analysis
**Duration:** 120-150 minutes  
**Difficulty:** ⭐⭐⭐⭐⭐

#### Objectives
- Combine FFT with computer vision concepts
- Implement scale-space representations
- Build foundation for feature detection algorithms

#### Tasks
1. **Laplacian pyramid construction:**
   ```python
   def build_laplacian_pyramid(image, levels=5):
       # Use frequency domain for efficient Gaussian filtering
       # Build Gaussian pyramid, then compute Laplacian levels
       pass
   ```

2. **Scale-space representation:**
   - Implement difference of Gaussians (DoG) using FFT
   - Create scale-space volume for blob detection
   - Connect to SIFT keypoint detection theory

3. **Image fusion applications:**
   - Multi-exposure fusion using pyramid blending
   - Focus stacking for extended depth of field
   - Seamless image compositing

4. **Frequency domain multi-scale analysis:**
   - Implement steerable pyramids
   - Compare with wavelet-based approaches
   - Analyze computational complexity

#### Expected Output
- Functions: `build_pyramid()`, `fuse_images()`, `detect_blobs()`
- Demonstrations: Multi-scale analysis visualizations
- Applications: Focus stacking or HDR-like results

#### Key Insights to Gain
- Multi-scale analysis is fundamental to computer vision
- Frequency domain enables efficient pyramid construction
- Connection between classical and modern feature detection

---

### Assignment 10: Real-time Filtering System
**Duration:** 150-180 minutes  
**Difficulty:** ⭐⭐⭐⭐⭐

#### Objectives
- Optimize for performance and practical deployment
- Build complete filtering application
- Implement real-time processing techniques

#### Tasks
1. **Real-time video filtering:**
   ```python
   def real_time_video_filter():
       cap = cv2.VideoCapture(0)
       while True:
           ret, frame = cap.read()
           # Apply FFT-based filtering with optimization
           filtered_frame = optimized_frequency_filter(frame)
           cv2.imshow('Filtered', filtered_frame)
   ```

2. **Optimization techniques:**
   - Pre-compute filter kernels
   - Use overlap-add for streaming data
   - Implement multi-threading for pipeline processing
   - Memory management for large images

3. **Interactive parameter tuning:**
   - Build GUI with sliders for filter parameters
   - Real-time visualization of frequency responses
   - A/B comparison functionality

4. **Performance benchmarking:**
   - Measure frames per second for different filter types
   - Compare with OpenCV implementations
   - Profile bottlenecks and optimize

5. **Complete filtering toolkit:**
   - Command-line interface for batch processing
   - Configuration file support
   - Multiple input/output format support
   - Logging and error handling

#### Expected Output
- Application: Complete filtering software with GUI
- Documentation: User manual and API documentation
- Benchmarks: Performance comparison report
- Demo: Video showing real-time filtering capabilities

#### Key Insights to Gain
- Real-world deployment requires careful optimization
- User interface design for technical applications
- Performance profiling and optimization techniques

---

## Implementation Guidelines

### Development Environment Setup
```bash
# Create conda environment
conda create -n signal_processing python=3.9
conda activate signal_processing

# Install required packages
pip install numpy scipy matplotlib opencv-python
pip install librosa soundfile  # for audio processing
pip install jupyter ipywidgets  # for interactive notebooks
pip install plotly  # for interactive plots
pip install tkinter  # for GUI applications
```

### Project Structure
```
fft_assignments/
├── assignment_01_fft_fundamentals/
├── assignment_02_frequency_analysis/
├── assignment_03_convolution_theorem/
├── assignment_04_filter_design/
├── assignment_05_audio_processing/
├── assignment_06_2d_fft_analysis/
├── assignment_07_image_filtering/
├── assignment_08_advanced_techniques/
├── assignment_09_multiscale_analysis/
├── assignment_10_realtime_system/
├── utils/
│   ├── visualization.py
│   ├── signal_generators.py
│   └── performance_utils.py
└── data/
    ├── audio_samples/
    ├── test_images/
    └── results/
```

### Success Metrics for Each Assignment

#### Technical Implementation (60%)
- Code runs without errors
- Outputs match expected results
- Proper handling of edge cases
- Code is well-documented and readable

#### Conceptual Understanding (25%)
- Can explain what the code does and why
- Connects implementation to theoretical concepts
- Identifies key insights and limitations
- Makes connections to computer vision applications

#### Innovation and Extension (15%)
- Goes beyond minimum requirements
- Implements optimizations or variations
- Explores additional applications
- Demonstrates creative problem-solving

### Study Schedule Recommendation

**Week 1 Plan (20 hours total):**
- **Day 1 (3 hours):** Assignments 1-2 (Foundation)
- **Day 2 (3 hours):** Assignment 3 + start Assignment 4
- **Day 3 (3 hours):** Complete Assignment 4 + Assignment 5
- **Day 4 (3 hours):** Assignments 6-7 (2D processing)
- **Day 5 (4 hours):** Assignment 8 (Advanced techniques)
- **Day 6-7 (4 hours):** Assignments 9-10 (Expert level)

### Debugging and Troubleshooting

#### Common Issues and Solutions

1. **FFT scaling issues:**
   - Remember FFT vs IFFT scaling conventions
   - Check if using fftshift/ifftshift correctly
   - Verify Parseval's theorem for energy conservation

2. **Circular convolution artifacts:**
   - Ensure proper zero-padding for linear convolution
   - Length should be len(x) + len(h) - 1
   - Use np.fft.fftconvolve for automatic handling

3. **2D image processing issues:**
   - Handle different image formats (uint8 vs float)
   - Remember to convert back to valid range [0,255] or [0,1]
   - Account for color vs grayscale images

4. **Performance bottlenecks:**
   - Use np.fft instead of implementing DFT for large signals
   - Consider using scipy.fft for additional optimizations
   - Profile code to identify actual bottlenecks

### Connection to Computer Vision

Each assignment builds toward computer vision applications:

- **Assignments 1-3:** Foundation for understanding convolution in CNNs
- **Assignments 4-5:** Preprocessing and feature extraction
- **Assignments 6-8:** Classical image processing for computer vision
- **Assignment 9:** Multi-scale analysis (SIFT, DoG, pyramids)
- **Assignment 10:** Real-time processing for vision systems

### Additional Resources

#### Documentation
- [NumPy FFT Tutorial](https://numpy.org/doc/stable/reference/routines.fft.html)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [OpenCV Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)

#### Interactive Learning
- [Jupyter widgets](https://ipywidgets.readthedocs.io/) for parameter exploration
- [Plotly](https://plotly.com/python/) for interactive frequency domain visualization
- [Bokeh](https://bokeh.org/) for real-time plotting applications

#### Advanced Topics
- [PyWavelets](https://pywavelets.readthedocs.io/) for wavelet analysis comparison
- [scikit-image](https://scikit-image.org/) for additional image processing functions
- [OpenCV contrib modules](https://github.com/opencv/opencv_contrib) for advanced algorithms

### Assessment and Next Steps

Upon completion of all assignments:

1. **Self-Assessment Questions:**
   - Can you explain when to use FFT vs spatial domain filtering?
   - Do you understand the trade-offs in filter design?
   - Can you connect these concepts to CNN operations?
   - Are you comfortable with 2D frequency domain analysis?

2. **Portfolio Development:**
   - Create a showcase of your best results
   - Document lessons learned and insights gained
   - Prepare code samples for technical interviews
   - Build foundation for advanced computer vision topics

3. **Transition to Module 4:**
   - You should now have solid foundation in signal processing
   - Ready to tackle classical computer vision algorithms
   - Prepared to understand CNN convolution operations
   - Equipped with practical filtering and preprocessing skills

This comprehensive guide provides everything needed to master FFT and signal processing concepts through hands-on implementation, setting a strong foundation for your computer vision career transition.
