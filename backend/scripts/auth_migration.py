import os
import sys
import psycopg2
from dotenv import load_dotenv

# Ensure we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
if not DB_URL:
    print("Error: SUPABASE_DB_URL not set in .env")
    sys.exit(1)

def migrate_db():
    print("Connecting to Supabase...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # Drop public.users
        print("Dropping public.users table...")
        cur.execute("DROP TABLE IF EXISTS users CASCADE;")
        
        # Truncate tables to remove old data
        print("Truncating old data...")
        cur.execute("TRUNCATE TABLE history CASCADE;")
        cur.execute("TRUNCATE TABLE jobs CASCADE;")
        cur.execute("TRUNCATE TABLE chatbot_feedback CASCADE;")

        # history table
        print("Updating history table...")
        cur.execute("DROP POLICY IF EXISTS \"Users see own history\" ON history;")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='history' AND column_name='user_id';")
        if not cur.fetchone():
            cur.execute("ALTER TABLE history ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;")
        else:
            cur.execute("ALTER TABLE history ALTER COLUMN user_id TYPE UUID USING user_id::uuid;")
            # Add foreign key constraint if not exists
            try:
                cur.execute("ALTER TABLE history ADD CONSTRAINT history_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;")
            except psycopg2.errors.DuplicateObject:
                pass
            
        cur.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id);")
            
        cur.execute("ALTER TABLE history ENABLE ROW LEVEL SECURITY;")
        cur.execute("""
            CREATE POLICY "Users see own history" 
            ON history FOR ALL 
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
        """)

        # jobs table
        print("Updating jobs table...")
        cur.execute("DROP POLICY IF EXISTS \"Users see own jobs\" ON jobs;")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='jobs' AND column_name='user_id';")
        if not cur.fetchone():
            cur.execute("ALTER TABLE jobs ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;")
        else:
            cur.execute("ALTER TABLE jobs ALTER COLUMN user_id TYPE UUID USING user_id::uuid;")
            try:
                cur.execute("ALTER TABLE jobs ADD CONSTRAINT jobs_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;")
            except psycopg2.errors.DuplicateObject:
                pass
                
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);")
            
        cur.execute("ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;")
        cur.execute("""
            CREATE POLICY "Users see own jobs" 
            ON jobs FOR ALL 
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
        """)

        # chatbot_feedback table
        print("Updating chatbot_feedback table...")
        cur.execute("DROP POLICY IF EXISTS \"Users see own feedback\" ON chatbot_feedback;")
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='chatbot_feedback' AND column_name='user_id';")
        if not cur.fetchone():
            cur.execute("ALTER TABLE chatbot_feedback ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;")
        else:
            cur.execute("ALTER TABLE chatbot_feedback ALTER COLUMN user_id TYPE UUID USING user_id::uuid;")
            try:
                cur.execute("ALTER TABLE chatbot_feedback ADD CONSTRAINT chatbot_feedback_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;")
            except psycopg2.errors.DuplicateObject:
                pass
                
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chatbot_feedback_user_id ON chatbot_feedback(user_id);")
            
        cur.execute("ALTER TABLE chatbot_feedback ENABLE ROW LEVEL SECURITY;")
        cur.execute("""
            CREATE POLICY "Users see own feedback" 
            ON chatbot_feedback FOR ALL 
            USING (auth.uid() = user_id)
            WITH CHECK (auth.uid() = user_id);
        """)
        
    conn.close()
    print("Database migration completed successfully!")

if __name__ == "__main__":
    migrate_db()
