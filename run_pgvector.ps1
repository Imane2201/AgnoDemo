# PowerShell script to run PgVector Docker container
# Run this script instead of the multi-line docker command

docker run -d `
    -e POSTGRES_DB=ai `
    -e POSTGRES_USER=ai `
    -e POSTGRES_PASSWORD=ai `
    -e PGDATA=/var/lib/postgresql/data/pgdata `
    -v pgvolume:/var/lib/postgresql/data `
    -p 5532:5432 `
    --name pgvector `
    agnohq/pgvector:16

Write-Host "PgVector container started successfully!"
Write-Host "You can check the status with: docker ps" 