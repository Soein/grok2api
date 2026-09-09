package account

// RecoveryResult reports one maintenance attempt without exposing upstream
// response bodies or credentials. Reason is a fixed, non-sensitive code.
type RecoveryResult struct {
	Claimed   bool
	Recovered bool
	Skipped   bool
	Reason    string
}
