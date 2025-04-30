# LLM Thought Analyzer

## Overview

This project is a tool for analyzing the thought process of LLMs. It uses a combination of techniques to extract the thought process from the LLM's output and visualize it in a graph.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

And run the following command to start the server:

```bash
streamlit run app.py
```

We've prepared a few sample files to test the app. You can find them in the `data/test_output/converted` folder.

## Usage

The app will load the JSON file and display the graph. You can then select a node to see the details of the node.

