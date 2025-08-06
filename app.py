from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import numpy as np
from scipy.fftpack import dct, idct
import pywt
from pydub import AudioSegment
import zlib
import tempfile
import uuid
import threading
import time
from werkzeug.utils import secure_filename
from mutagen.flac import FLAC
from mutagen.aiff import AIFF
import base64
import io


app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    UPLOAD_FOLDER = '/tmp'  # Render utilise /tmp pour les fichiers temporaires

app.config.from_object(Config)

# Dictionnaire pour stocker les tâches en cours
active_tasks = {}

LOSSLESS_FORMATS = ["wav", "flac", "aiff"]
LOSSY_FORMATS = ["mp3", "aac", "ogg", "wma", "m4a", "opus"]

SUPPORTED_EXTENSIONS = ['wav', 'flac', 'aiff', 'mp3', 'ogg', 'aac', 'm4a', 'opus', 'wma']

# Page HTML pour l'interface utilisateur
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Watermarking</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"], input[type="number"], input[type="file"], select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 4px;
            display: none;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background-color: #eee;
            border: 1px solid #ddd;
            flex: 1;
            text-align: center;
        }
        .tab.active {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .progress-container {
            width: 100%;
            background-color: #f3f3f3;
            border-radius: 4px;
            margin: 15px 0;
            display: none;
        }
        .progress-bar {
            height: 20px;
            background-color: #4CAF50;
            border-radius: 4px;
            width: 0%;
            text-align: center;
            line-height: 20px;
            color: white;
        }
        .advanced-toggle {
            text-align: center;
            margin: 10px 0;
            cursor: pointer;
            color: #3498db;
        }
        .advanced-options {
            display: none;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
        }
        @media (max-width: 600px) {
            body {
                padding: 10px;
            }
            .container {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <h1>Audio Watermarking</h1>
    
    <div class="tabs">
        <div class="tab active" id="embed-tab">Insérer un watermark</div>
        <div class="tab" id="extract-tab">Extraire un watermark</div>
    </div>
    
    <div class="tab-content active" id="embed-content">
        <div class="container">
            <form id="embed-form">
                <div class="form-group">
                    <label for="audio-file-embed">Fichier audio:</label>
                    <input type="file" id="audio-file-embed" name="audio_file" accept=".wav,.flac,.aiff,.mp3,.ogg,.aac,.m4a,.opus,.wma">
                </div>
                
                <div class="form-group">
                    <label for="watermark-text">Texte du watermark (max 12 caractères):</label>
                    <input type="text" id="watermark-text" name="watermark_text" maxlength="12" required>
                </div>
                
                <div class="form-group">
                    <label for="method-embed">Méthode:</label>
                    <select id="method-embed" name="method">
                        <option value="DCT">DCT</option>
                        <option value="DWT-DCT">DWT-DCT</option>
                    </select>
                </div>
                
                <div class="advanced-toggle" id="advanced-toggle-embed">
                    Afficher les options avancées ▼
                </div>
                
                <div class="advanced-options" id="advanced-options-embed">
                    <div class="form-group">
                        <label for="segment-length">Longueur de segment:</label>
                        <input type="number" id="segment-length" name="segment_length" value="2048" min="512" max="16384" step="512">
                    </div>
                    
                    <div class="form-group">
                        <label for="seed">Seed:</label>
                        <input type="number" id="seed" name="seed" value="42" min="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="modulation-strength">Force de modulation:</label>
                        <input type="number" id="modulation-strength" name="modulation_strength" value="0.005" min="0.001" max="0.5" step="0.001">
                    </div>
                    
                    <div class="form-group">
                        <label for="band-lower">Bande de fréquence inférieure (%):</label>
                        <input type="number" id="band-lower" name="band_lower_pct" value="33" min="1" max="90">
                    </div>
                    
                    <div class="form-group">
                        <label for="band-upper">Bande de fréquence supérieure (%):</label>
                        <input type="number" id="band-upper" name="band_upper_pct" value="66" min="10" max="99">
                    </div>
                    
                    <div class="form-group">
                        <label for="n-coeffs">Nombre de coefficients:</label>
                        <input type="number" id="n-coeffs" name="n_coeffs" value="5" min="1" max="20">
                    </div>
                    
                    <div class="dwt-options" style="display: none;">
                        <div class="form-group">
                            <label for="dwt-level">Niveau DWT:</label>
                            <input type="number" id="dwt-level" name="dwt_level" value="1" min="1" max="5">
                        </div>
                        
                        <div class="form-group">
                            <label for="dwt-wavelet">Ondelette:</label>
                            <select id="dwt-wavelet" name="dwt_wavelet">
                                <option value="haar">haar</option>
                                <!-- Les autres ondelettes seront chargées dynamiquement -->
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="dwt-coeff-type">Type de coefficient:</label>
                            <select id="dwt-coeff-type" name="dwt_coeff_type">
                                <option value="cA">cA (Approximation)</option>
                                <option value="cD">cD (Détail)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <button type="submit">Insérer le watermark</button>
            </form>
            
            <div class="progress-container" id="progress-container-embed">
                <div class="progress-bar" id="progress-bar-embed">0%</div>
            </div>
            
            <div class="result" id="result-embed">
                <h3>Résultat:</h3>
                <p id="result-text-embed"></p>
                <div id="download-container"></div>
            </div>
        </div>
    </div>
    
    <div class="tab-content" id="extract-content">
        <div class="container">
            <form id="extract-form">
                <div class="form-group">
                    <label for="audio-file-extract">Fichier audio avec watermark:</label>
                    <input type="file" id="audio-file-extract" name="audio_file" accept=".wav,.flac,.aiff,.mp3,.ogg,.aac,.m4a,.opus,.wma">
                </div>
                
                <div class="form-group">
                    <label for="watermark-length">Longueur attendue du watermark:</label>
                    <input type="number" id="watermark-length" name="watermark_length" value="12" min="1" max="64">
                </div>
                
                <div class="form-group">
                    <label for="method-extract">Méthode:</label>
                    <select id="method-extract" name="method">
                        <option value="DCT">DCT</option>
                        <option value="DWT-DCT">DWT-DCT</option>
                    </select>
                </div>
                
                <div class="advanced-toggle" id="advanced-toggle-extract">
                    Afficher les options avancées ▼
                </div>
                
                <div class="advanced-options" id="advanced-options-extract">
                    <div class="form-group">
                        <label for="segment-length-extract">Longueur de segment:</label>
                        <input type="number" id="segment-length-extract" name="segment_length" value="2048" min="512" max="16384" step="512">
                    </div>
                    
                    <div class="form-group">
                        <label for="seed-extract">Seed:</label>
                        <input type="number" id="seed-extract" name="seed" value="42" min="1">
                    </div>
                    
                    <div class="form-group">
                        <label for="modulation-strength-extract">Force de modulation:</label>
                        <input type="number" id="modulation-strength-extract" name="modulation_strength" value="0.005" min="0.001" max="0.5" step="0.001">
                    </div>
                    
                    <div class="form-group">
                        <label for="band-lower-extract">Bande de fréquence inférieure (%):</label>
                        <input type="number" id="band-lower-extract" name="band_lower_pct" value="33" min="1" max="90">
                    </div>
                    
                    <div class="form-group">
                        <label for="band-upper-extract">Bande de fréquence supérieure (%):</label>
                        <input type="number" id="band-upper-extract" name="band_upper_pct" value="66" min="10" max="99">
                    </div>
                    
                    <div class="form-group">
                        <label for="n-coeffs-extract">Nombre de coefficients:</label>
                        <input type="number" id="n-coeffs-extract" name="n_coeffs" value="5" min="1" max="20">
                    </div>
                    
                    <div class="dwt-options-extract" style="display: none;">
                        <div class="form-group">
                            <label for="dwt-level-extract">Niveau DWT:</label>
                            <input type="number" id="dwt-level-extract" name="dwt_level" value="1" min="1" max="5">
                        </div>
                        
                        <div class="form-group">
                            <label for="dwt-wavelet-extract">Ondelette:</label>
                            <select id="dwt-wavelet-extract" name="dwt_wavelet">
                                <option value="haar">haar</option>
                                <!-- Les autres ondelettes seront chargées dynamiquement -->
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="dwt-coeff-type-extract">Type de coefficient:</label>
                            <select id="dwt-coeff-type-extract" name="dwt_coeff_type">
                                <option value="cA">cA (Approximation)</option>
                                <option value="cD">cD (Détail)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <button type="submit">Extraire le watermark</button>
            </form>
            
            <div class="progress-container" id="progress-container-extract">
                <div class="progress-bar" id="progress-bar-extract">0%</div>
            </div>
            
            <div class="result" id="result-extract">
                <h3>Watermark extrait:</h3>
                <p id="result-text-extract"></p>
            </div>
        </div>
    </div>

    <script>
        // Fonction pour charger les ondelettes disponibles
        function loadWavelets() {
            fetch('/api/wavelets')
                .then(response => response.json())
                .then(data => {
                    const wavelets = data.wavelets;
                    const selectEmbed = document.getElementById('dwt-wavelet');
                    const selectExtract = document.getElementById('dwt-wavelet-extract');
                    
                    // Vider les sélecteurs
                    selectEmbed.innerHTML = '';
                    selectExtract.innerHTML = '';
                    
                    // Ajouter les ondelettes
                    wavelets.forEach(wavelet => {
                        const optionEmbed = document.createElement('option');
                        optionEmbed.value = wavelet;
                        optionEmbed.textContent = wavelet;
                        selectEmbed.appendChild(optionEmbed);
                        
                        const optionExtract = document.createElement('option');
                        optionExtract.value = wavelet;
                        optionExtract.textContent = wavelet;
                        selectExtract.appendChild(optionExtract);
                    });
                })
                .catch(error => console.error('Erreur lors du chargement des ondelettes:', error));
        }
        
        // Chargement des ondelettes au démarrage
        document.addEventListener('DOMContentLoaded', loadWavelets);
        
        // Gestion des onglets
        document.getElementById('embed-tab').addEventListener('click', function() {
            document.getElementById('embed-tab').classList.add('active');
            document.getElementById('extract-tab').classList.remove('active');
            document.getElementById('embed-content').classList.add('active');
            document.getElementById('extract-content').classList.remove('active');
        });
        
        document.getElementById('extract-tab').addEventListener('click', function() {
            document.getElementById('extract-tab').classList.add('active');
            document.getElementById('embed-tab').classList.remove('active');
            document.getElementById('extract-content').classList.add('active');
            document.getElementById('embed-content').classList.remove('active');
        });
        
        // Gestion des options avancées
        document.getElementById('advanced-toggle-embed').addEventListener('click', function() {
            const advancedOptions = document.getElementById('advanced-options-embed');
            if (advancedOptions.style.display === 'block') {
                advancedOptions.style.display = 'none';
                this.textContent = 'Afficher les options avancées ▼';
            } else {
                advancedOptions.style.display = 'block';
                this.textContent = 'Masquer les options avancées ▲';
            }
        });
        
        document.getElementById('advanced-toggle-extract').addEventListener('click', function() {
            const advancedOptions = document.getElementById('advanced-options-extract');
            if (advancedOptions.style.display === 'block') {
                advancedOptions.style.display = 'none';
                this.textContent = 'Afficher les options avancées ▼';
            } else {
                advancedOptions.style.display = 'block';
                this.textContent = 'Masquer les options avancées ▲';
            }
        });
        
        // Afficher/masquer les options DWT en fonction de la méthode
        document.getElementById('method-embed').addEventListener('change', function() {
            const dwtOptions = document.querySelector('.dwt-options');
            if (this.value === 'DWT-DCT') {
                dwtOptions.style.display = 'block';
            } else {
                dwtOptions.style.display = 'none';
            }
        });
        
        document.getElementById('method-extract').addEventListener('change', function() {
            const dwtOptions = document.querySelector('.dwt-options-extract');
            if (this.value === 'DWT-DCT') {
                dwtOptions.style.display = 'block';
            } else {
                dwtOptions.style.display = 'none';
            }
        });
        
        // Fonction pour suivre la progression d'une tâche
        function trackTaskProgress(taskId, progressBar, resultDiv, callback) {
            const interval = setInterval(() => {
                fetch(`/api/task/${taskId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'completed') {
                            clearInterval(interval);
                            progressBar.style.width = '100%';
                            progressBar.textContent = '100%';
                            if (callback) callback(data);
                        } else if (data.status === 'error') {
                            clearInterval(interval);
                            progressBar.style.width = '100%';
                            progressBar.style.backgroundColor = '#f44336';
                            progressBar.textContent = 'Erreur';
                            resultDiv.style.display = 'block';
                            document.getElementById('result-text-embed').textContent = data.error || 'Une erreur est survenue';
                        } else {
                            const progress = data.progress || 0;
                            progressBar.style.width = `${progress}%`;
                            progressBar.textContent = `${progress}%`;
                        }
                    })
                    .catch(error => {
                        console.error('Erreur lors du suivi de la tâche:', error);
                    });
            }, 1000);
        }
        
        // Formulaire d'insertion de watermark
        document.getElementById('embed-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const progressContainer = document.getElementById('progress-container-embed');
            const progressBar = document.getElementById('progress-bar-embed');
            const resultDiv = document.getElementById('result-embed');
            
            // Afficher la barre de progression
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            resultDiv.style.display = 'none';
            
            // Envoyer la requête
            fetch('/api/embed', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Suivre la progression
                trackTaskProgress(data.task_id, progressBar, resultDiv, function(taskData) {
                    resultDiv.style.display = 'block';
                    document.getElementById('result-text-embed').textContent = `Watermark inséré avec succès. Force de modulation finale: ${data.final_modulation}`;
                    
                    // Créer un lien de téléchargement
                    const downloadContainer = document.getElementById('download-container');
                    downloadContainer.innerHTML = '';
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `data:audio/${data.filename.split('.').pop()};base64,${data.file_data}`;
                    downloadLink.download = data.filename;
                    downloadLink.textContent = 'Télécharger le fichier audio avec watermark';
                    downloadLink.style.display = 'block';
                    downloadLink.style.marginTop = '10px';
                    downloadLink.classList.add('button');
                    downloadLink.style.textDecoration = 'none';
                    downloadLink.style.backgroundColor = '#4CAF50';
                    downloadLink.style.color = 'white';
                    downloadLink.style.padding = '10px 15px';
                    downloadLink.style.borderRadius = '4px';
                    downloadLink.style.textAlign = 'center';
                    downloadContainer.appendChild(downloadLink);
                });
            })
            .catch(error => {
                progressBar.style.width = '100%';
                progressBar.style.backgroundColor = '#f44336';
                progressBar.textContent = 'Erreur';
                resultDiv.style.display = 'block';
                document.getElementById('result-text-embed').textContent = error.message || 'Une erreur est survenue';
            });
        });
        
        // Formulaire d'extraction de watermark
        document.getElementById('extract-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const progressContainer = document.getElementById('progress-container-extract');
            const progressBar = document.getElementById('progress-bar-extract');
            const resultDiv = document.getElementById('result-extract');
            
            // Afficher la barre de progression
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            resultDiv.style.display = 'none';
            
            // Envoyer la requête
            fetch('/api/extract', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Suivre la progression
                trackTaskProgress(data.task_id, progressBar, resultDiv, function(taskData) {
                    resultDiv.style.display = 'block';
                    document.getElementById('result-text-extract').textContent = data.extracted_watermark || 'Aucun watermark détecté';
                });
            })
            .catch(error => {
                progressBar.style.width = '100%';
                progressBar.style.backgroundColor = '#f44336';
                progressBar.textContent = 'Erreur';
                resultDiv.style.display = 'block';
                document.getElementById('result-text-extract').textContent = error.message || 'Une erreur est survenue';
            });
        });
    </script>
</body>
</html>
"""

def get_audio_format(filepath):
    ext = os.path.splitext(filepath)[1].lower().replace('.', '')
    return ext if ext else None

def is_lossless(fmt):
    return fmt in LOSSLESS_FORMATS

def cleanup_file(filepath, delay=60):  # Réduit le délai à 1 minute
    """Nettoie un fichier après un délai plus court pour Render"""
    def delayed_cleanup():
        time.sleep(delay)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
    threading.Thread(target=delayed_cleanup, daemon=True).start()

class AudioWatermarker:
    def __init__(self):
        self.segment_length = 2048
        self.seed = 42
        self.modulation_strength = 0.005
        self.band_lower_pct = 33
        self.band_upper_pct = 66
        self.dwt_level = 1
        self.dwt_wavelet = 'haar'
        self.dwt_coeff_type = 'cA'
        self.n_coeffs = 5

    def audio_to_numpy(self, audio_source):
        fmt = get_audio_format(audio_source)
        if fmt is None:
            raise ValueError("Format de fichier non supporté.")
        audio_seg = AudioSegment.from_file(audio_source, format=fmt)
        if audio_seg.channels > 1:
            audio_seg = audio_seg.set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
        samples /= 32768.0
        return samples, audio_seg.frame_rate, fmt

    def numpy_to_audio_bytes(self, samples, sample_rate, fmt):
        samples = np.clip(samples, -1, 1)
        samples = (samples * 32768).astype(np.int16)
        audio_seg = AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        buffer = io.BytesIO()
        export_kwargs = {}
        if fmt in LOSSY_FORMATS:
            export_kwargs["bitrate"] = "320k"
        audio_seg.export(buffer, format=fmt, **export_kwargs)
        return buffer.getvalue()

    def numpy_to_audio(self, samples, sample_rate, output_path, fmt):
        samples = np.clip(samples, -1, 1)
        samples = (samples * 32768).astype(np.int16)
        audio_seg = AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        export_kwargs = {}
        if fmt in LOSSY_FORMATS:
            export_kwargs["bitrate"] = "320k"
        audio_seg.export(output_path, format=fmt, **export_kwargs)

    def audio_bytes_to_numpy(self, audio_bytes, fmt):
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmpfile:
            tmp_path = tmpfile.name
            tmpfile.write(audio_bytes)
        samples, sr, _ = self.audio_to_numpy(tmp_path)
        os.remove(tmp_path)
        return samples, sr

    # -------------- Hamming 7,4 --------------
    def hamming_encode_bitblock(self, nibble):
        if len(nibble) != 4:
            raise ValueError("La taille du bloc doit être de 4 bits")
        d1, d2, d3, d4 = nibble
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        return [p1, p2, d1, p3, d2, d3, d4]

    def hamming_decode_bitblock(self, block):
        if len(block) != 7:
            raise ValueError("La taille du bloc doit être de 7 bits")
        p1, p2, d1, p3, d2, d3, d4 = block
        c1 = p1 ^ d1 ^ d2 ^ d4
        c2 = p2 ^ d1 ^ d3 ^ d4
        c3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = (c1 * 1) + (c2 * 2) + (c3 * 4)
        if syndrome > 0:
            if syndrome == 1:    p1 = 1 - p1
            elif syndrome == 2:  p2 = 1 - p2
            elif syndrome == 3:  d1 = 1 - d1
            elif syndrome == 4:  p3 = 1 - p3
            elif syndrome == 5:  d2 = 1 - d2
            elif syndrome == 6:  d3 = 1 - d3
            elif syndrome == 7:  d4 = 1 - d4
        return [d1, d2, d3, d4]

    def hamming_encode_bitstring(self, bit_list):
        if len(bit_list) % 4 != 0:
            padding = 4 - (len(bit_list) % 4)
            bit_list.extend([0] * padding)
        encoded = []
        for i in range(0, len(bit_list), 4):
            nibble = bit_list[i:i+4]
            encoded.extend(self.hamming_encode_bitblock(nibble))
        return encoded

    def hamming_decode_bitstring(self, encoded_bits):
        if len(encoded_bits) % 7 != 0:
            raise ValueError("La longueur des bits encodés doit être un multiple de 7")
        decoded = []
        for i in range(0, len(encoded_bits), 7):
            block = encoded_bits[i:i+7]
            if len(block) == 7:
                decoded.extend(self.hamming_decode_bitblock(block))
        return decoded

    def get_coeff_indices(self, band_lower, band_upper, n, key):
        rng = np.random.RandomState(key)
        indices = rng.choice(np.arange(band_lower, band_upper), size=n, replace=False)
        return indices

    def _auto_segment_length(self, audio_len, bits_needed):
        # Cherche la plus petite puissance de 2 qui permet de caser bits_needed dans audio_len/segment_length
        for seglen in [512, 1024, 2048, 4096, 8192, 16384]:
            if (audio_len // seglen) >= bits_needed:
                return seglen
        # Si vraiment trop court, prend le max possible
        return max(128, audio_len // bits_needed)

    def embed_watermark(self, audio, watermark, segment_length=None, seed=None, modulation_strength=None):
        segment_length = segment_length or self.segment_length
        seed = seed or self.seed
        modulation_strength = modulation_strength or self.modulation_strength
        n_coeffs = self.n_coeffs

        watermark_bytes = watermark.encode('utf-8')
        crc = zlib.crc32(watermark_bytes).to_bytes(4, 'big')
        watermark_with_crc = watermark_bytes + crc
        wm_bits = []
        for byte in watermark_with_crc:
            for i in range(7, -1, -1):
                wm_bits.append(1 if (byte >> i) & 1 else 0)
        encoded_bits = self.hamming_encode_bitstring(wm_bits)
        rep_length = len(encoded_bits)
        num_segments = len(audio) // segment_length
        redundancy = num_segments // rep_length

        if redundancy < 1:
            # Watermark direct (sans CRC ni Hamming)
            wm_bits = []
            for byte in watermark_bytes:
                for i in range(7, -1, -1):
                    wm_bits.append(1 if (byte >> i) & 1 else 0)
            bits_needed = len(wm_bits)
            segment_length = self._auto_segment_length(len(audio), bits_needed)
            num_segments = len(audio) // segment_length
            if num_segments < bits_needed:
                raise ValueError("Le signal est trop court pour contenir le watermark, même sans correction d'erreur.")
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=bits_needed, replace=False)
            band_lower = int(segment_length * self.band_lower_pct / 100)
            band_upper = int(segment_length * self.band_upper_pct / 100)
            audio_watermarked = np.copy(audio)
            for bit_idx, (seg_idx, bit) in enumerate(zip(segment_indices, wm_bits)):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = dct(segment, norm='ortho')
                coeff_indices = self.get_coeff_indices(band_lower, band_upper, n_coeffs, bit_idx + seed)
                for idx in coeff_indices:
                    coeffs[idx] += modulation_strength * (1 if bit == 1 else -1)
                audio_watermarked[start:end] = idct(coeffs, norm='ortho').astype(np.float32)
            return audio_watermarked
        else:
            watermark_bits_full = encoded_bits * redundancy
            watermark_mod = [1 if bit == 1 else -1 for bit in watermark_bits_full]
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=len(watermark_mod), replace=False)
            band_lower = int(segment_length * self.band_lower_pct / 100)
            band_upper = int(segment_length * self.band_upper_pct / 100)
            audio_watermarked = np.copy(audio)
            for bit_idx, (seg_idx, bit) in enumerate(zip(segment_indices, watermark_mod)):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = dct(segment, norm='ortho')
                coeff_indices = self.get_coeff_indices(band_lower, band_upper, n_coeffs, bit_idx + seed)
                for idx in coeff_indices:
                    coeffs[idx] += modulation_strength * bit
                audio_watermarked[start:end] = idct(coeffs, norm='ortho').astype(np.float32)
            return audio_watermarked

    def extract_watermark(self, audio, watermark_message_length, segment_length=None, seed=None, modulation_strength=None):
        segment_length = segment_length or self.segment_length
        seed = seed or self.seed
        n_coeffs = self.n_coeffs

        message_bytes_length = watermark_message_length + 4
        bits_per_byte = 8
        total_data_bits = message_bytes_length * bits_per_byte
        hamming_blocks_needed = (total_data_bits + 3) // 4
        rep_length = hamming_blocks_needed * 7
        num_segments = len(audio) // segment_length
        redundancy = num_segments // rep_length

        if redundancy < 1:
            bits_needed = watermark_message_length * 8
            segment_length = self._auto_segment_length(len(audio), bits_needed)
            num_segments = len(audio) // segment_length
            if num_segments < bits_needed:
                raise ValueError("Le signal est trop court pour extraire le watermark.")
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=bits_needed, replace=False)
            band_lower = int(segment_length * self.band_lower_pct / 100)
            band_upper = int(segment_length * self.band_upper_pct / 100)
            bits = []
            for bit_idx, seg_idx in enumerate(segment_indices):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = dct(segment, norm='ortho')
                coeff_indices = self.get_coeff_indices(band_lower, band_upper, n_coeffs, bit_idx + seed)
                votes = [1 if coeffs[idx] >= 0 else -1 for idx in coeff_indices]
                bits.append(1 if np.sum(votes) >= 0 else 0)
            bytes_array = []
            for i in range(0, len(bits), 8):
                byte_bits = bits[i:i+8]
                if len(byte_bits) < 8:
                    break
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                bytes_array.append(byte_val)
            watermark_bytes = bytes(bytes_array)
            try:
                watermark_str = watermark_bytes.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                watermark_str = "Erreur de décodage (UTF-8, court)"
            return watermark_str
        else:
            total_bits = redundancy * rep_length
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=total_bits, replace=False)
            band_lower = int(segment_length * self.band_lower_pct / 100)
            band_upper = int(segment_length * self.band_upper_pct / 100)
            extracted_bits = []
            for bit_idx, seg_idx in enumerate(segment_indices):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = dct(segment, norm='ortho')
                coeff_indices = self.get_coeff_indices(band_lower, band_upper, n_coeffs, bit_idx + seed)
                votes = [1 if coeffs[idx] >= 0 else -1 for idx in coeff_indices]
                extracted_bits.append(1 if np.sum(votes) >= 0 else 0)
            arr = np.array(extracted_bits).reshape((redundancy, rep_length))
            final_bits = []
            for col in range(rep_length):
                final_bits.append(1 if np.sum(arr[:, col]) >= (redundancy / 2.0) else 0)
            decoded_bits = self.hamming_decode_bitstring(final_bits)
            decoded_bits = decoded_bits[:total_data_bits]
            bytes_array = []
            for i in range(0, len(decoded_bits), 8):
                byte_bits = decoded_bits[i:i+8]
                if len(byte_bits) < 8: break
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                bytes_array.append(byte_val)
            watermark_bytes = bytes(bytes_array)
            if len(watermark_bytes) < 4:
                raise ValueError("Watermark décodé trop court.")
            message = watermark_bytes[:-4]
            crc_extracted = watermark_bytes[-4:]
            crc_calculated = zlib.crc32(message).to_bytes(4, 'big')
            try:
                watermark_str = message.decode('utf-8')
            except UnicodeDecodeError:
                watermark_str = "Erreur de décodage (UTF-8)"
            if crc_extracted != crc_calculated:
                print("Attention : CRC non vérifié, le watermark extrait peut être incorrect.")
            else:
                print("CRC vérifié avec succès.")
            return watermark_str

    def embed_watermark_dwt_dct(self, audio, watermark, segment_length=None, seed=None, modulation_strength=None, dwt_level=None, dwt_wavelet=None, dwt_coeff_type=None):
        segment_length = segment_length or self.segment_length
        seed = seed or self.seed
        modulation_strength = modulation_strength or self.modulation_strength
        n_coeffs = self.n_coeffs
        dwt_level = dwt_level if dwt_level is not None else self.dwt_level
        dwt_wavelet = dwt_wavelet if dwt_wavelet is not None else self.dwt_wavelet
        dwt_coeff_type = dwt_coeff_type if dwt_coeff_type is not None else self.dwt_coeff_type

        watermark_bytes = watermark.encode('utf-8')
        crc = zlib.crc32(watermark_bytes).to_bytes(4, 'big')
        watermark_with_crc = watermark_bytes + crc
        wm_bits = []
        for byte in watermark_with_crc:
            for i in range(7, -1, -1):
                wm_bits.append(1 if (byte >> i) & 1 else 0)
        encoded_bits = self.hamming_encode_bitstring(wm_bits)
        rep_length = len(encoded_bits)
        num_segments = len(audio) // segment_length
        redundancy = num_segments // rep_length

        if redundancy < 1:
            wm_bits = []
            for byte in watermark_bytes:
                for i in range(7, -1, -1):
                    wm_bits.append(1 if (byte >> i) & 1 else 0)
            bits_needed = len(wm_bits)
            segment_length = self._auto_segment_length(len(audio), bits_needed)
            num_segments = len(audio) // segment_length
            if num_segments < bits_needed:
                raise ValueError("Le signal est trop court pour contenir le watermark, même sans correction d'erreur.")
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=bits_needed, replace=False)
            audio_watermarked = np.copy(audio)
            for bit_idx, (seg_idx, bit) in enumerate(zip(segment_indices, wm_bits)):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = pywt.wavedec(segment, dwt_wavelet, level=dwt_level)
                if dwt_coeff_type == 'cA':
                    c = coeffs[0]
                else:
                    c = coeffs[1]
                idx_low = int(len(c) * self.band_lower_pct / 100)
                idx_up = int(len(c) * self.band_upper_pct / 100)
                coeff_indices = self.get_coeff_indices(idx_low, idx_up, n_coeffs, bit_idx + seed + 123)
                c_mod = dct(c, norm='ortho')
                for idx in coeff_indices:
                    c_mod[idx] += modulation_strength * (1 if bit == 1 else -1)
                c = idct(c_mod, norm='ortho')
                if dwt_coeff_type == 'cA':
                    coeffs[0] = c
                else:
                    coeffs[1] = c
                segment_mod = pywt.waverec(coeffs, dwt_wavelet)
                segment_mod = segment_mod[:segment_length]
                audio_watermarked[start:end] = segment_mod.astype(np.float32)
            return audio_watermarked
        else:
            watermark_bits_full = encoded_bits * redundancy
            watermark_mod = [1 if bit == 1 else -1 for bit in watermark_bits_full]
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=len(watermark_mod), replace=False)
            audio_watermarked = np.copy(audio)
            for bit_idx, (seg_idx, bit) in enumerate(zip(segment_indices, watermark_mod)):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = pywt.wavedec(segment, dwt_wavelet, level=dwt_level)
                if dwt_coeff_type == 'cA':
                    c = coeffs[0]
                else:
                    c = coeffs[1]
                idx_low = int(len(c) * self.band_lower_pct / 100)
                idx_up = int(len(c) * self.band_upper_pct / 100)
                coeff_indices = self.get_coeff_indices(idx_low, idx_up, n_coeffs, bit_idx + seed + 123)
                c_mod = dct(c, norm='ortho')
                for idx in coeff_indices:
                    c_mod[idx] += modulation_strength * bit
                c = idct(c_mod, norm='ortho')
                if dwt_coeff_type == 'cA':
                    coeffs[0] = c
                else:
                    coeffs[1] = c
                segment_mod = pywt.waverec(coeffs, dwt_wavelet)
                segment_mod = segment_mod[:segment_length]
                audio_watermarked[start:end] = segment_mod.astype(np.float32)
            return audio_watermarked

    def extract_watermark_dwt_dct(self, audio, watermark_message_length, segment_length=None, seed=None, modulation_strength=None, dwt_level=None, dwt_wavelet=None, dwt_coeff_type=None):
        segment_length = segment_length or self.segment_length
        seed = seed or self.seed
        n_coeffs = self.n_coeffs
        dwt_level = dwt_level if dwt_level is not None else self.dwt_level
        dwt_wavelet = dwt_wavelet if dwt_wavelet is not None else self.dwt_wavelet
        dwt_coeff_type = dwt_coeff_type if dwt_coeff_type is not None else self.dwt_coeff_type

        message_bytes_length = watermark_message_length + 4
        bits_per_byte = 8
        total_data_bits = message_bytes_length * bits_per_byte
        hamming_blocks_needed = (total_data_bits + 3) // 4
        rep_length = hamming_blocks_needed * 7
        num_segments = len(audio) // segment_length
        redundancy = num_segments // rep_length

        if redundancy < 1:
            bits_needed = watermark_message_length * 8
            segment_length = self._auto_segment_length(len(audio), bits_needed)
            num_segments = len(audio) // segment_length
            if num_segments < bits_needed:
                raise ValueError("Le signal est trop court pour extraire le watermark.")
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=bits_needed, replace=False)
            bits = []
            for bit_idx, seg_idx in enumerate(segment_indices):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = pywt.wavedec(segment, dwt_wavelet, level=dwt_level)
                if dwt_coeff_type == 'cA':
                    c = coeffs[0]
                else:
                    c = coeffs[1]
                idx_low = int(len(c) * self.band_lower_pct / 100)
                idx_up = int(len(c) * self.band_upper_pct / 100)
                coeff_indices = self.get_coeff_indices(idx_low, idx_up, n_coeffs, bit_idx + seed + 123)
                c_mod = dct(c, norm='ortho')
                votes = [1 if c_mod[idx] >= 0 else -1 for idx in coeff_indices]
                bits.append(1 if np.sum(votes) >= 0 else 0)
            bytes_array = []
            for i in range(0, len(bits), 8):
                byte_bits = bits[i:i+8]
                if len(byte_bits) < 8:
                    break
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                bytes_array.append(byte_val)
            watermark_bytes = bytes(bytes_array)
            try:
                watermark_str = watermark_bytes.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                watermark_str = "Erreur de décodage (UTF-8, court)"
            return watermark_str
        else:
            total_bits = redundancy * rep_length
            np.random.seed(seed)
            segment_indices = np.random.choice(np.arange(num_segments), size=total_bits, replace=False)
            extracted_bits = []
            for bit_idx, seg_idx in enumerate(segment_indices):
                start = seg_idx * segment_length
                end = start + segment_length
                segment = audio[start:end]
                segment = np.array(segment, dtype=np.float64)
                coeffs = pywt.wavedec(segment, dwt_wavelet, level=dwt_level)
                if dwt_coeff_type == 'cA':
                    c = coeffs[0]
                else:
                    c = coeffs[1]
                idx_low = int(len(c) * self.band_lower_pct / 100)
                idx_up = int(len(c) * self.band_upper_pct / 100)
                coeff_indices = self.get_coeff_indices(idx_low, idx_up, n_coeffs, bit_idx + seed + 123)
                c_mod = dct(c, norm='ortho')
                votes = [1 if c_mod[idx] >= 0 else -1 for idx in coeff_indices]
                extracted_bits.append(1 if np.sum(votes) >= 0 else 0)
            arr = np.array(extracted_bits).reshape((redundancy, rep_length))
            final_bits = []
            for col in range(rep_length):
                final_bits.append(1 if np.sum(arr[:, col]) >= (redundancy / 2.0) else 0)
            decoded_bits = self.hamming_decode_bitstring(final_bits)
            decoded_bits = decoded_bits[:total_data_bits]
            bytes_array = []
            for i in range(0, len(decoded_bits), 8):
                byte_bits = decoded_bits[i:i+8]
                if len(byte_bits) < 8: break
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                bytes_array.append(byte_val)
            watermark_bytes = bytes(bytes_array)
            if len(watermark_bytes) < 4:
                raise ValueError("Watermark décodé trop court.")
            message = watermark_bytes[:-4]
            crc_extracted = watermark_bytes[-4:]
            crc_calculated = zlib.crc32(message).to_bytes(4, 'big')
            try:
                watermark_str = message.decode('utf-8')
            except UnicodeDecodeError:
                watermark_str = "Erreur de décodage (UTF-8)"
            if crc_extracted != crc_calculated:
                print("Attention : CRC non vérifié, le watermark extrait peut être incorrect.")
            else:
                print("CRC vérifié avec succès.")
            return watermark_str

    def embed_watermark_with_test(self, audio, watermark, segment_length=None, seed=None, modulation_strength=None, fmt="wav", method="DCT", dwt_level=None, dwt_wavelet=None, dwt_coeff_type=None):
        segment_length = segment_length or self.segment_length
        seed = seed or self.seed
        current_modulation = modulation_strength if modulation_strength is not None else self.modulation_strength
        max_modulation = 0.5
        watermark_fixed = watermark.ljust(12)[:12]
        sample_rate = 44100
        if isinstance(audio, tuple) and len(audio) >= 2:
            audio, sample_rate = audio[:2]
        embed_func = self.embed_watermark if method == "DCT" else self.embed_watermark_dwt_dct
        extract_func = self.extract_watermark if method == "DCT" else self.extract_watermark_dwt_dct

        while current_modulation <= max_modulation:
            print(f"Tentative avec modulation_strength = {current_modulation}")
            watermarked_audio = embed_func(
                audio, watermark_fixed, segment_length, seed, current_modulation,
                dwt_level, dwt_wavelet, dwt_coeff_type
            ) if method != "DCT" else embed_func(
                audio, watermark_fixed, segment_length, seed, current_modulation
            )
            audio_bytes = self.numpy_to_audio_bytes(watermarked_audio, sample_rate, fmt)
            test_audio, _ = self.audio_bytes_to_numpy(audio_bytes, fmt)
            try:
                extracted = extract_func(
                    test_audio, len(watermark_fixed), segment_length, seed, current_modulation,
                    dwt_level, dwt_wavelet, dwt_coeff_type
                ) if method != "DCT" else extract_func(
                    test_audio, len(watermark_fixed), segment_length, seed, current_modulation
                )
            except Exception as e:
                extracted = ""
            if extracted.strip() == watermark_fixed.strip():
                print(f"Watermark inséré avec succès avec modulation_strength = {current_modulation}")
                return watermarked_audio, current_modulation
            else:
                print(f"Échec avec modulation_strength = {current_modulation}, augmentation de 0.005")
                current_modulation += 0.005
        raise ValueError("Impossible d'insérer correctement le watermark dans les limites de modulation.")

    def pad_lossless(self, input_path, output_path, fmt):
        orig_size = os.path.getsize(input_path)
        new_size = os.path.getsize(output_path)
        if new_size >= orig_size:
            return
        diff = orig_size - new_size
        if fmt == "wav":
            with open(output_path, "ab") as f:
                f.write(b"\x00" * diff)
        elif fmt == "flac":
            try:
                f = FLAC(output_path)
                f["WATERMARK_PADDING"] = "x" * diff
                f.save()
            except Exception:
                with open(output_path, "ab") as f:
                    f.write(b"\x00" * diff)
        elif fmt == "aiff":
            try:
                aiff = AIFF(output_path)
                aiff["comment"] = [" " * diff]
                aiff.save()
            except Exception:
                with open(output_path, "ab") as f:
                    f.write(b"\x00" * diff)

# Instance globale du watermarker
watermarker = AudioWatermarker()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/embed', methods=['POST'])
def embed_watermark():
    try:
        task_id = str(uuid.uuid4())
        active_tasks[task_id] = {"status": "processing", "progress": 0}
        
        # Récupération des paramètres
        if 'audio_file' not in request.files:
            return jsonify({"error": "Aucun fichier audio fourni"}), 400
            
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400
            
        watermark_text = request.form.get('watermark_text', '')
        if not watermark_text:
            return jsonify({"error": "Texte du watermark requis"}), 400
            
        if len(watermark_text) > 12:
            return jsonify({"error": "Le texte du watermark ne peut dépasser 12 caractères"}), 400
            
        # Paramètres optionnels
        method = request.form.get('method', 'DCT')
        segment_length = int(request.form.get('segment_length', watermarker.segment_length))
        seed = int(request.form.get('seed', watermarker.seed))
        modulation_strength = float(request.form.get('modulation_strength', watermarker.modulation_strength))
        band_lower_pct = float(request.form.get('band_lower_pct', watermarker.band_lower_pct))
        band_upper_pct = float(request.form.get('band_upper_pct', watermarker.band_upper_pct))
        dwt_level = int(request.form.get('dwt_level', watermarker.dwt_level))
        dwt_wavelet = request.form.get('dwt_wavelet', watermarker.dwt_wavelet)
        dwt_coeff_type = request.form.get('dwt_coeff_type', watermarker.dwt_coeff_type)
        n_coeffs = int(request.form.get('n_coeffs', watermarker.n_coeffs))
        
        # Sauvegarde temporaire du fichier
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{get_audio_format(file.filename)}")
        file.save(temp_input.name)
        temp_input.close()
        
        active_tasks[task_id]["progress"] = 20
        
        # Configuration du watermarker
        watermarker.segment_length = segment_length
        watermarker.seed = seed
        watermarker.modulation_strength = modulation_strength
        watermarker.band_lower_pct = band_lower_pct
        watermarker.band_upper_pct = band_upper_pct
        watermarker.dwt_level = dwt_level
        watermarker.dwt_wavelet = dwt_wavelet
        watermarker.dwt_coeff_type = dwt_coeff_type
        watermarker.n_coeffs = n_coeffs
        
        # Traitement audio
        audio, sample_rate, fmt_in = watermarker.audio_to_numpy(temp_input.name)
        fmt_out = get_audio_format(file.filename)
        
        active_tasks[task_id]["progress"] = 40
        
        watermark_fixed = watermark_text.ljust(12)[:12]
        
        # Insertion du watermark
        if method == "DCT":
            watermarked_audio, final_modulation = watermarker.embed_watermark_with_test(
                (audio, sample_rate), watermark_fixed, segment_length, seed, 
                modulation_strength, fmt_out, method
            )
        else:
            watermarked_audio, final_modulation = watermarker.embed_watermark_with_test(
                (audio, sample_rate), watermark_fixed, segment_length, seed, 
                modulation_strength, fmt_out, method, dwt_level, dwt_wavelet, dwt_coeff_type
            )
        
        active_tasks[task_id]["progress"] = 80
        
        # Génération du fichier de sortie
        output_bytes = watermarker.numpy_to_audio_bytes(watermarked_audio, sample_rate, fmt_out)
        
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["status"] = "completed"
        
        # Nettoyage du fichier temporaire d'entrée
        cleanup_file(temp_input.name, 10)
        
        # Retour du fichier encodé en base64
        encoded_file = base64.b64encode(output_bytes).decode('utf-8')
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "file_data": encoded_file,
            "filename": f"{os.path.splitext(file.filename)[0]}_watermarked{os.path.splitext(file.filename)[1]}",
            "final_modulation": final_modulation
        })
        
    except Exception as e:
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "error"
            active_tasks[task_id]["error"] = str(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def extract_watermark():
    try:
        task_id = str(uuid.uuid4())
        active_tasks[task_id] = {"status": "processing", "progress": 0}
        
        # Récupération des paramètres
        if 'audio_file' not in request.files:
            return jsonify({"error": "Aucun fichier audio fourni"}), 400
            
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({"error": "Aucun fichier sélectionné"}), 400
            
        watermark_length = int(request.form.get('watermark_length', 12))
        
        # Paramètres optionnels
        method = request.form.get('method', 'DCT')
        segment_length = int(request.form.get('segment_length', watermarker.segment_length))
        seed = int(request.form.get('seed', watermarker.seed))
        modulation_strength = float(request.form.get('modulation_strength', watermarker.modulation_strength))
        band_lower_pct = float(request.form.get('band_lower_pct', watermarker.band_lower_pct))
        band_upper_pct = float(request.form.get('band_upper_pct', watermarker.band_upper_pct))
        dwt_level = int(request.form.get('dwt_level', watermarker.dwt_level))
        dwt_wavelet = request.form.get('dwt_wavelet', watermarker.dwt_wavelet)
        dwt_coeff_type = request.form.get('dwt_coeff_type', watermarker.dwt_coeff_type)
        n_coeffs = int(request.form.get('n_coeffs', watermarker.n_coeffs))
        
        # Sauvegarde temporaire du fichier
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{get_audio_format(file.filename)}")
        file.save(temp_input.name)
        temp_input.close()
        
        active_tasks[task_id]["progress"] = 30
        
        # Configuration du watermarker
        watermarker.segment_length = segment_length
        watermarker.seed = seed
        watermarker.modulation_strength = modulation_strength
        watermarker.band_lower_pct = band_lower_pct
        watermarker.band_upper_pct = band_upper_pct
        watermarker.dwt_level = dwt_level
        watermarker.dwt_wavelet = dwt_wavelet
        watermarker.dwt_coeff_type = dwt_coeff_type
        watermarker.n_coeffs = n_coeffs
        
        # Traitement audio
        audio, _, _ = watermarker.audio_to_numpy(temp_input.name)
        
        active_tasks[task_id]["progress"] = 60
        
        # Extraction du watermark
        if method == "DCT":
            extracted_watermark = watermarker.extract_watermark(
                audio, watermark_length, segment_length, seed, modulation_strength
            )
        else:
            extracted_watermark = watermarker.extract_watermark_dwt_dct(
                audio, watermark_length, segment_length, seed, modulation_strength,
                dwt_level, dwt_wavelet, dwt_coeff_type
            )
        
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["status"] = "completed"
        
        # Nettoyage du fichier temporaire
        cleanup_file(temp_input.name, 10)
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "extracted_watermark": extracted_watermark
        })
        
    except Exception as e:
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "error"
            active_tasks[task_id]["error"] = str(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    if task_id in active_tasks:
        return jsonify(active_tasks[task_id])
    return jsonify({"error": "Tâche non trouvée"}), 404

@app.route('/api/wavelets')
def get_wavelets():
    wavelets = [w for w in pywt.wavelist(kind='discrete') if not w.startswith('bior') and not w.startswith('rbio')]
    return jsonify({"wavelets": wavelets})

# Point de terminaison pour le health check
@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Utiliser le port défini par l'environnement ou 5000 par défaut
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
