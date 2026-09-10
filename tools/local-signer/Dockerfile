ARG RUNTIME_IMAGE=alpine:3.23
FROM golang:1.26-alpine AS builder
WORKDIR /src
COPY backend/go.mod backend/go.sum ./
COPY backend/cmd/grok-signer ./cmd/grok-signer
COPY backend/internal/pkg/localstatsig ./internal/pkg/localstatsig
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /grok-signer ./cmd/grok-signer
FROM ${RUNTIME_IMAGE}
COPY --from=builder /grok-signer /usr/local/bin/grok-signer
COPY backend/internal/pkg/localstatsig/LICENSE /usr/share/licenses/grok-signer/LICENSE
USER 10001:10001
ENTRYPOINT ["/usr/local/bin/grok-signer"]
CMD ["--listen", "0.0.0.0:8788", "--config", "/signer/params.json"]
