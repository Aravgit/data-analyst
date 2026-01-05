# Frontend service container
FROM node:20-alpine AS builder

WORKDIR /app

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

FROM node:20-alpine AS runner

ENV NODE_ENV=production
WORKDIR /app

COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./next.config.js

EXPOSE 3001

CMD ["npm", "run", "start", "--", "--port=3001"]
