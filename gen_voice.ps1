
Add-Type -AssemblyName System.Speech
 = New-Object System.Speech.Synthesis.SpeechSynthesizer
.SetOutputToWaveFile('sample_data/test_voice.wav')
.Speak('Testing spoken audio generation for clinical dementia evaluation.')
.Dispose()
