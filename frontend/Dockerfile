# Tablero de métricas. Multi-stage: se compila con Node y se sirve con nginx, así que
# la imagen final no lleva node_modules ni el toolchain de build.

# --- Etapa 1: build ---------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY index.html vite.config.js ./
COPY public ./public
COPY src ./src

RUN npm run build


# --- Etapa 2: servir --------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/99-config-ach.sh

RUN chmod +x /docker-entrypoint.d/99-config-ach.sh

EXPOSE 80
