import axios from 'axios'

/** Base Axios instance.  All requests go to /api/* which is proxied to Flask. */
const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.error ?? err.message ?? 'Unknown error'
    return Promise.reject(new Error(msg))
  },
)

export default api
