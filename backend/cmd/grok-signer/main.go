// grok-signer serves local challenge signatures from a private, expiring
// snapshot. It does not obtain credentials or contact any upstream service.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/chenyme/grok2api/backend/internal/pkg/localstatsig"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8788", "HTTP listen address (private network only)")
	config := flag.String("config", "", "private JSON snapshot path; containing directory must be mounted for atomic updates")
	flag.Parse()
	if *config == "" {
		log.Fatal("--config is required")
	}
	server := &http.Server{Addr: *listen, Handler: localstatsig.NewHandler(*config), ReadHeaderTimeout: 2 * time.Second, ReadTimeout: 5 * time.Second, WriteTimeout: 5 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 8192}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Print("local signer starting; readiness validates configuration only")
	serveErr := make(chan error, 1)
	go func() { serveErr <- server.ListenAndServe() }()
	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal("signer HTTP server failed")
		}
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			log.Print("signer graceful shutdown timed out")
		}
	}
}
