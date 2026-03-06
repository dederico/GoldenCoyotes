# Configuración de PostgreSQL en Render para Persistencia

## Problema Actual
SQLite en Render free tier **NO persiste datos** entre deployments porque el sistema de archivos es efímero. Cada vez que se hace deploy, la base de datos se reinicia.

## Solución: PostgreSQL en Render

### Paso 1: Crear base de datos PostgreSQL en Render

1. Ir a [Render Dashboard](https://dashboard.render.com/)
2. Click en "New +" → "PostgreSQL"
3. Configurar:
   - **Name:** `golden-coyotes-db`
   - **Database:** `golden_coyotes`
   - **User:** `golden_coyotes_user`
   - **Region:** Oregon (US West) - mismo que el web service
   - **Plan:** Free
4. Click "Create Database"
5. Esperar a que se cree (toma ~2-3 minutos)
6. Copiar el **Internal Database URL** que aparece en la página

### Paso 2: Configurar variables de entorno en el Web Service

1. Ir a tu Web Service en Render
2. Settings → Environment
3. Agregar variable de entorno:
   ```
   DATABASE_URL = [pegar el Internal Database URL]
   ```
   Ejemplo:
   ```
   DATABASE_URL = postgresql://golden_coyotes_user:XXX@dpg-XXX/golden_coyotes
   ```

### Paso 3: Actualizar `requirements.txt`

Agregar dependencias de PostgreSQL:

```txt
psycopg2-binary==2.9.9
```

### Paso 4: Actualizar `database_setup.py` (Ya implementado)

El código ya detecta automáticamente si existe `DATABASE_URL` y usa PostgreSQL.

```python
import os

# Auto-detect database type
if os.getenv('DATABASE_URL'):
    # PostgreSQL en producción
    db_url = os.getenv('DATABASE_URL')
else:
    # SQLite en desarrollo
    db_path = "golden_coyotes.db"
```

### Paso 5: Redeploy

1. Commit y push cambios:
   ```bash
   git add .
   git commit -m "feat: soporte postgresql para persistencia"
   git push origin main
   ```

2. Render detectará automáticamente el push y desplegará

3. Durante el deploy, las migraciones se ejecutarán automáticamente

### Verificación

1. Ir a `https://goldencoyotes.onrender.com`
2. Registrar un usuario de prueba
3. Crear una oportunidad
4. Hacer un nuevo deploy (modificar cualquier archivo y push)
5. Verificar que los datos **persisten** después del deploy

## Beneficios

✅ **Persistencia real:** Los datos sobreviven entre deploys
✅ **Mejor rendimiento:** PostgreSQL es más rápido que SQLite para operaciones concurrentes
✅ **Escalabilidad:** Preparado para producción
✅ **Backups automáticos:** Render hace backups diarios (en planes pagos)

## Limitaciones del Plan Free

- **Storage:** 1 GB
- **Expire:** Base de datos se elimina después de 90 días de inactividad
- **Connections:** Máximo 97 conexiones concurrentes
- **No backups:** El plan free no incluye backups automáticos

## Migración de SQLite a PostgreSQL (Si ya tienes datos)

Si ya tienes datos en SQLite local que quieres migrar:

```bash
# 1. Exportar datos de SQLite
sqlite3 golden_coyotes.db .dump > backup.sql

# 2. Convertir a PostgreSQL (manualmente o con herramienta)
# Herramienta recomendada: pgloader

# 3. Importar a PostgreSQL
psql $DATABASE_URL < backup_pg.sql
```

## Troubleshooting

### Error: "relation does not exist"
- Las tablas no se crearon. Verificar que `init_database()` se ejecutó.
- Revisar logs de Render para ver errores de migración.

### Error: "could not connect to server"
- Verificar que `DATABASE_URL` está correctamente configurada
- Verificar que el web service y la BD están en la misma región

### Datos no persisten
- Verificar que `DATABASE_URL` está configurada (no en blanco)
- Verificar en logs que dice "Using PostgreSQL" y no "Using SQLite"

## Alternativas (Si no quieres PostgreSQL)

1. **Supabase:** Base de datos PostgreSQL gratis con mejor plan free
2. **Railway:** Similar a Render pero con mejor plan free
3. **MongoDB Atlas:** Base de datos NoSQL con plan free generoso
4. **PlanetScale:** MySQL serverless con plan free

---

**Nota importante:** Una vez configurado PostgreSQL, el seed de datos demo (`/admin/seed-demo-data`) se debe ejecutar UNA SOLA VEZ después del primer deploy para poblar la base de datos.
