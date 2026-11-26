-- One-time Supabase setup for the satellite fog layer.
-- Creates a public-read Storage bucket. Writes need the service-role key
-- (only the GitHub Actions worker has it); the backend reads the public URL.
insert into storage.buckets (id, name, public)
values ('fog-satellite', 'fog-satellite', true)
on conflict (id) do update set public = true;
