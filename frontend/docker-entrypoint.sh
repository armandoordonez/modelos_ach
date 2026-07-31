#!/bin/sh
# Escribe la configuración de arranque del tablero.
#
# La URL de la API se inyecta aquí y no en tiempo de build: cambiar de entorno
# (local, staging, producción) no obliga a reconstruir la imagen. Aquí no se escribe
# ninguna credencial: el navegador solo habla con el backend.
#
# nginx ejecuta todos los scripts de /docker-entrypoint.d/ antes de arrancar, así que
# este archivo solo hace su trabajo y termina.

set -e

API_URL="${ACH_API_URL:-http://localhost:8000}"
cat > /usr/share/nginx/html/config.js <<EOF
window.__ACH_CONFIG__ = { apiUrl: "${API_URL}" };
EOF

echo "Tablero configurado contra la API en ${API_URL}"
