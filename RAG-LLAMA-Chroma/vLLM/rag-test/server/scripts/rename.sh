#!/bin/bash

# Caminho inicial (atual por padrão)
DIR="${1:-.}"

# Percorre recursivamente todos os arquivos e adiciona .txt se não tiver extensão
find "$DIR" -type f | while read -r file; do
  # Verifica se o arquivo não tem extensão
  if [[ "$file" != *.* ]]; then
    mv "$file" "${file}.txt"
    echo "Renomeado: $file → ${file}.txt"
  fi
done

