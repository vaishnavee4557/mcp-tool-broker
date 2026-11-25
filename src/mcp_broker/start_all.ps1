$ErrorActionPreference = "Stop"

$PROJECT = "C:\Users\vaish\Downloads\mcp_tool_broker"

Write-Host ""
Write-Host "========================================"
Write-Host " FACTIGENT - START ALL"
Write-Host "========================================"
Write-Host ""

Set-Location $PROJECT

# -------------------------------------------------
# 1. Activate Python environment
# -------------------------------------------------

Write-Host "[1/8] Activating Python environment..."

& "$PROJECT\.venv\Scripts\Activate.ps1"

$env:PYTHONPATH = (Resolve-Path "$PROJECT\src").Path


# -------------------------------------------------
# 2. Check Docker
# -------------------------------------------------

Write-Host "[2/8] Checking Docker..."

docker info *> $null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is not running."
    Write-Host "Start Docker Desktop and run this script again."
    exit 1
}

Write-Host "Docker OK"


# -------------------------------------------------
# 3. Start PostgreSQL / pgvector
# -------------------------------------------------

Write-Host "[3/8] Starting PostgreSQL pgvector..."

$container = docker ps -a `
    --filter "name=factigent_pgvector" `
    --format "{{.Names}}"

if ($container -eq "factigent_pgvector") {

    $running = docker ps `
        --filter "name=factigent_pgvector" `
        --format "{{.Names}}"

    if ($running -ne "factigent_pgvector") {
        docker start factigent_pgvector
    }

}
else {

    docker run -d `
        --name factigent_pgvector `
        -e POSTGRES_USER=factigent_app `
        -e POSTGRES_PASSWORD=factigent_dev_password `
        -e POSTGRES_DB=factigent_chatbot `
        -p 5433:5432 `
        pgvector/pgvector:pg17
}

Write-Host "PostgreSQL/pgvector OK"


# -------------------------------------------------
# 4. Check Play + UMS backend
# -------------------------------------------------

Write-Host "[4/8] Checking Play/UMS backend on port 5001..."

$backend = Test-NetConnection `
    localhost `
    -Port 5001 `
    -WarningAction SilentlyContinue

if (-not $backend.TcpTestSucceeded) {

    Write-Host ""
    Write-Host "ERROR:"
    Write-Host "Play/UMS backend is NOT running on localhost:5001"
    Write-Host ""
    Write-Host "Start your Java/Spring Boot backend first."
    Write-Host ""
    exit 1
}

Write-Host "Play/UMS backend OK"


# -------------------------------------------------
# 5. Start MCP 8001 - Predictive
# -------------------------------------------------

Write-Host "[5/8] Starting Predictive MCP :8001..."

Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", @"
cd '$PROJECT'
& '.\.venv\Scripts\Activate.ps1'
`$env:PYTHONPATH = (Resolve-Path '.\src').Path
`$env:OPENAPI_SOURCE_NAME='predictive_maintenance'
`$env:MCP_PORT='8001'
python -m mcp_broker.mcp_server
"@


# -------------------------------------------------
# 6. Start MCP 8002 - Play
# -------------------------------------------------

Write-Host "[6/8] Starting Play MCP :8002..."

Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", @"
cd '$PROJECT'
& '.\.venv\Scripts\Activate.ps1'
`$env:PYTHONPATH = (Resolve-Path '.\src').Path
`$env:OPENAPI_SOURCE_NAME='play'
`$env:MCP_PORT='8002'
python -m mcp_broker.mcp_server
"@


# -------------------------------------------------
# 7. Start MCP 8003 - UMS
# -------------------------------------------------

Write-Host "[7/8] Starting UMS MCP :8003..."

Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", @"
cd '$PROJECT'
& '.\.venv\Scripts\Activate.ps1'
`$env:PYTHONPATH = (Resolve-Path '.\src').Path
`$env:OPENAPI_SOURCE_NAME='ums'
`$env:MCP_PORT='8003'
python -m mcp_broker.mcp_server
"@


# Give MCP servers time to start
Write-Host ""
Write-Host "Waiting for MCP servers..."
Start-Sleep -Seconds 8


# -------------------------------------------------
# 8. Sync + Embed + CLI
# -------------------------------------------------

Write-Host "[8/8] Syncing tools..."

python -m mcp_broker.sync_tools

Write-Host ""
Write-Host "Embedding new tools..."

python -m mcp_broker.embed_tools


Write-Host ""
Write-Host "========================================"
Write-Host " ALL SERVICES STARTED"
Write-Host "========================================"
Write-Host ""
Write-Host "Predictive MCP : http://127.0.0.1:8001/mcp"
Write-Host "Play MCP       : http://127.0.0.1:8002/mcp"
Write-Host "UMS MCP        : http://127.0.0.1:8003/mcp"
Write-Host ""
Write-Host "Starting chatbot CLI..."
Write-Host ""

python -m mcp_broker.cli 