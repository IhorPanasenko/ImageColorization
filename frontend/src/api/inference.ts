import api from './client'
import type { ColorizeResult, ColorizeMode, ModelType } from '@/types'

export const inferenceApi = {
  colorize: (
    file: File,
    model: ModelType,
    checkpoint: string,
    mode: ColorizeMode,
    referenceFile?: File,
  ) => {
    const form = new FormData()
    form.append('file', file)
    form.append('model', model)
    form.append('checkpoint', checkpoint)
    form.append('mode', mode)
    if (referenceFile) {
      form.append('reference_file', referenceFile)
    }
    return api.post<ColorizeResult>('', form, {
      baseURL: '/api/colorize',
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180_000,
    }).then((r) => r.data)
  },

  colorizeBatch: (
    files: File[],
    model: ModelType,
    checkpoint: string,
    mode: ColorizeMode,
    referenceFile?: File,
  ) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    form.append('model', model)
    form.append('checkpoint', checkpoint)
    form.append('mode', mode)
    if (referenceFile) {
      form.append('reference_file', referenceFile)
    }
    return api.post<(ColorizeResult & { filename: string })[]>(
      '/colorize/batch',
      form,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 180_000,
      },
    ).then((r) => r.data)
  },
}
