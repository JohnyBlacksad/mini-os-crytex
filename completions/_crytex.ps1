Register-ArgumentCompleter -Native -CommandName crytex,crytex-kernel -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $words = @(
        "doctor", "project", "index", "rag", "token-economy", "goal", "plan",
        "kanban", "run", "review", "diag", "models", "security", "prompts",
        "lora", "evolution", "bench", "sandbox", "backend-acceptance", "prove",
        "prove-release-gate", "--json", "--strict", "--report-path"
    )
    $words |
        Where-Object { $_ -like "$wordToComplete*" } |
        ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, "ParameterValue", $_)
        }
}
