/**
 * Owns the item list and every mutation on it.
 *
 * Ingestion is asynchronous, so the list is the client's view of a state
 * machine it does not control. The hook polls - but only while something is
 * actually in flight, so an idle tab makes no requests. Polling is the right
 * call at this size: one endpoint, one client, no socket lifecycle to manage.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api.js';

const POLL_INTERVAL_MS = 1500;

export const useItems = () => {
  const [items, setItems] = useState([]);
  const [counts, setCounts] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listItems();
      setItems(data.items);
      setCounts(data.counts);
      setError(null);
      return data.items;
    } catch (err) {
      setError(err);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  // Re-poll only while at least one item is still being processed.
  useEffect(() => {
    const inFlight = items.some((item) => item.status === 'pending' || item.status === 'processing');
    clearTimeout(timerRef.current);
    if (inFlight) {
      timerRef.current = setTimeout(refresh, POLL_INTERVAL_MS);
    }
    return () => clearTimeout(timerRef.current);
  }, [items, refresh]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const addNote = useCallback(
    async (content, title) => {
      const { item } = await api.ingestNote(content, title);
      // Optimistic insert so the pending row appears immediately; the poll
      // started by its `pending` status takes over from here.
      setItems((current) => [item, ...current]);
      return item;
    },
    [],
  );

  const addUrl = useCallback(async (url, title) => {
    const { item } = await api.ingestUrl(url, title);
    setItems((current) => [item, ...current]);
    return item;
  }, []);

  const retry = useCallback(async (id) => {
    const { item } = await api.retryItem(id);
    setItems((current) => current.map((existing) => (existing.id === id ? item : existing)));
  }, []);

  const remove = useCallback(async (id) => {
    await api.deleteItem(id);
    setItems((current) => current.filter((existing) => existing.id !== id));
  }, []);

  return { items, counts, loading, error, refresh, addNote, addUrl, retry, remove };
};
