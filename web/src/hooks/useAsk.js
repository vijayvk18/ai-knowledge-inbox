/**
 * Owns one question at a time.
 *
 * An in-flight request is aborted when a new question is asked, so a slow
 * answer cannot land after a newer one and overwrite it.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api.js';

export const useAsk = () => {
  const [result, setResult] = useState(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);
  const controllerRef = useRef(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const ask = useCallback(async (question, topK = 5) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setAsking(true);
    setError(null);
    try {
      const data = await api.ask(question, topK, controller.signal);
      setResult(data);
      return data;
    } catch (err) {
      if (err.name === 'AbortError') return null;
      setError(err);
      setResult(null);
      return null;
    } finally {
      if (controllerRef.current === controller) setAsking(false);
    }
  }, []);

  const clear = useCallback(() => {
    controllerRef.current?.abort();
    setResult(null);
    setError(null);
  }, []);

  return { result, asking, error, ask, clear };
};
