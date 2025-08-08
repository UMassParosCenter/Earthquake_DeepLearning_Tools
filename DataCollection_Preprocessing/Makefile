# Makefile for running earthquake training data pipeline

PYTHON := python3

# The default target
all: generate_background earthquake_data psd_background psd_earthquake

generate_background:
	$(PYTHON) generateBackgroundData.py

earthquake_data:
	$(PYTHON) usgsEarthquakeDataGrabber.py

psd_background:
	$(PYTHON) PSD_Background_processor.py

psd_earthquake:
	$(PYTHON) PSD_Earthquake_processor.py

# Clean target (optional)
clean:
	rm -f *.pyc
	rm -rf __pycache__

.PHONY: all generate_background earthquake_data psd_background psd_earthquake clean
